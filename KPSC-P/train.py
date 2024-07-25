import os, sys
import torch.nn as nn
from datasets import DATASETS
import torch.utils.data as Data
import torch.distributed as dist
from tqdm import tqdm
import time, random, logging
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import options
from models.actionclip_based_visual_transformer import visual_prompt
from models.verb_prompt_learner import clip_prompt_sentence_encoder
from models.losses import *
from utils.Augmentation import *
import clip_code
from clip_code import clip
from utils.tools import *
from utils.base_utils import *
from utils.solver import _optimizer, _lr_scheduler

# used for getting params of DDP
def setup_distributed(opt, backend="nccl", port=29490):
    """Initialize slurm distributed training environment. (from mmcv)"""
    num_gpus = torch.cuda.device_count()
    dist.init_process_group(backend=backend)
    opt.local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(opt.local_rank)
    # specify master port
    os.environ['PYTHONHASHSEED'] = str(2020)
    os.environ["MASTER_PORT"] = str(opt.port)
    os.environ["WORLD_SIZE"] = str(torch.distributed.get_world_size())
    os.environ["LOCAL_RANK"] = str(opt.local_rank)
    os.environ["RANK"] = str(opt.local_rank)



class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.encoder = model.visual

    def forward(self,image):
        return self.encoder(image)

def frozen(net):
    for name, parameter in net.named_parameters():
        parameter.requires_grad = False


def saving_models(working_dir, name, epoch, model_image, fusion_model, model_text, optimizer):
    save_path = os.path.join(working_dir, name)
    torch.save({
        'epoch': epoch,
        'model_image_state_dict': model_image.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'model_text_state_dict': model_text.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path, _use_new_zipfile_serialization = True)  # just change to your preferred folder/filename

# 1. set up distributed device
setup_seed(2022)
opt_class = options.BaseOptions()
opt = opt_class.parse()

if opt.is_distributed:
    setup_distributed(opt, backend="nccl", port=opt.port)
    #setup_distributed(backend="nccl", port=opt.port)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda")


# 2. inital opt and log and seed
if (not opt.is_distributed) or (dist.get_rank() == 0):
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)
    if not os.path.exists(opt.modeldir):
        os.makedirs(opt.modeldir)

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.basicConfig(filename=os.path.join(opt.logdir, opt.log_name), filemode='w', level=logging.DEBUG,
                                                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    root_logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(stdout_handler)
    opt_class.print_options(opt)

## 3. transform
transform_train = get_augmentation(True, opt)
transform_val = get_augmentation(False, opt)

if opt.randaug_N > 0:
    transform_train = randAugment(transform_train, opt)


## 4. loadding models and set models's state
clip_model, clip_state_dict = clip_code.load(opt.backbone_path, device='cpu', jit=False, tsm=opt.tsm, joint=opt.joint, T=opt.num_segments, dropout=opt.visual_drop_out, emb_dropout=opt.visual_emb_dropout, pretrain=True) #Must set jit=False for training  ViT-B/32
### visual models
model_image  = ImageCLIP(clip_model)
fusion_model = visual_prompt(opt.sim_header, clip_state_dict, opt.num_segments)
### sentence models
model_text = clip_prompt_sentence_encoder(opt, clip_model)

model_image = model_image.to(device)
fusion_model = fusion_model.to(device)
model_text = model_text.to(device)

if opt.is_distributed:
    model_image = torch.nn.parallel.DistributedDataParallel(
        model_image, device_ids=[local_rank], output_device=local_rank
    )
    fusion_model = torch.nn.parallel.DistributedDataParallel(
        fusion_model, device_ids=[local_rank], output_device=local_rank
    )
    model_text = torch.nn.parallel.DistributedDataParallel(
        model_text, device_ids=[local_rank], output_device=local_rank
    )
else:
    model_image  = torch.nn.DataParallel(model_image)
    fusion_model = torch.nn.DataParallel(fusion_model)
    model_text  = torch.nn.DataParallel(model_text)


### frozen net
if (not opt.is_distributed) or (dist.get_rank() == 0):
    logging.info('train transforms: {}'.format(transform_train.transforms))
    logging.info('val transforms: {}'.format(transform_val.transforms))

    if opt.frozen_clip:
        logging.info('-' * 114)
        logging.info(' ' * 20 + "Frozen Clip.")
        logging.info('-' * 114)
        # frozen image encoder
        frozen(clip_model)
        frozen(model_image)
        # frozen text embedding and postion. eg
        for name, parameter in model_text.named_parameters():
            if name != 'module.prompt_learner.ctx':
                parameter.requries_grad = False

    logging.info('=' * 114)
    logging.info(' ' * 20 + "Networks Intial State")
    logging.info('=' * 114)
    logging.info('-' * 114)
    logging.info(' ' * 20 + "model image")
    logging.info('-' * 114)
    for name, parameter in model_image.named_parameters():
        logging.info("{} {}".format(name, parameter.requires_grad))
    logging.info('-' * 114)
    logging.info(' ' * 20 + "fusion model")
    logging.info('-' * 114)
    for name, parameter in fusion_model.named_parameters():
        logging.info("{} {}".format(name, parameter.requires_grad))
    logging.info('-' * 114)
    logging.info(' ' * 20 + "model text")
    logging.info('-' * 114)
    for name, parameter in model_text.named_parameters():
        logging.info("{} {}".format(name, parameter.requires_grad))
    logging.info('=' * 114)

## 5. inital datasets
train_data = DATASETS(opt.train_json_file, opt.datasets_root_path ,num_segments=opt.num_segments, image_tmpl=opt.image_tmpl, random_shift=opt.random_shift,
                    transform=transform_train)
# train_loader = DataLoader(train_data, batch_size=opt.batchsize, num_workers=opt.workers, shuffle=True, pin_memory=False, drop_last=True)

if opt.is_distributed:
    train_sampler = torch.utils.data.DistributedSampler(
        train_data,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank())

    train_loader = Data.DataLoader(dataset=train_data, 
                                    batch_size=opt.batchsize, 
                                    sampler=train_sampler,
                                    num_workers=opt.workers,
                                    pin_memory=opt.pin_memo,
                                    drop_last=True)
else:
    train_loader = Data.DataLoader(dataset=train_data, 
                                    batch_size=opt.batchsize, 
                                    shuffle=True, 
                                    num_workers=opt.workers,
                                    pin_memory=False,
                                    drop_last=True)


# 6. define optimizer and lr_scheduler
optimizer = _optimizer(opt, model_image, model_text, fusion_model)
lr_scheduler = _lr_scheduler(opt, optimizer)


# 7. load pretrained model
if opt.load_pretrained:
    checkpoint = torch.load(opt.load_model_path)
    model_text.load_state_dict(checkpoint['model_text_state_dict'])
    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
    model_image.load_state_dict(checkpoint['model_image_state_dict'])
    del checkpoint
    if (not opt.is_distributed) or (dist.get_rank() == 0):
        logging.info('load over from: {}'.format(opt.load_model_path))


lowest_loss = opt.now_best

# 8. training
crossentropyloss = nn.CrossEntropyLoss()
epoch_iters = len(train_loader)
#ctx_dim = clip_model.ln_final.weight.shape[0]
#logit_scale = clip_model.logit_scale.exp()
#Contrast_loss = contrast_loss(opt, logit_scale, ctx_dim)

for epoch in range(opt.start_epoch, opt.epochs):
    model_image.train()
    model_text.train()
    fusion_model.train()
    epoch_loss = 0
    period_loss = 0
    period_cnt = 0
    period_time = time.time()
    start_time = time.time()
    for id, (img_input, nouns_input, nouns_numbers) in enumerate(train_loader):
        
        optimizer.zero_grad()
        img_input = img_input.to(device)
        nouns_input = nouns_input.to(device)
        nouns_numbers = nouns_numbers.to(device)

        img_input = img_input.view((-1, opt.num_segments, 3) + img_input.size()[-2:])#[32, 8, 3, 224, 224]
        b, t, c, h, w = img_input.size()
        img_input = img_input.view(-1, c, h, w)

        image_embedding = model_image(img_input)
        image_embedding = image_embedding.view(b, t, -1)
        image_embedding = fusion_model(image_embedding)

        text_embedding = model_text(nouns_input, nouns_numbers)

        labels = torch.arange(opt.batchsize).to(device)
        #final_loss = Contrast_loss(image_embedding, text_embedding, labels)

        if opt.n_verb == 1:
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
            loss_imgs = crossentropyloss(logits_per_image, labels)
            loss_texts = crossentropyloss(logits_per_text, labels)
            final_loss = (loss_imgs + loss_texts) / 2.0
        else:
            # select the hgihest verb logits as final logits
            ctx_dim = clip_model.ln_final.weight.shape[0]
            logit_scale = clip_model.logit_scale.exp()
            text_embedding = text_embedding.reshape(opt.batchsize ,opt.n_verb, ctx_dim)
            logits_per_image, logits_per_text = create_multiverbs_logits(image_embedding, text_embedding, logit_scale)
            loss_imgs = crossentropyloss(logits_per_image, labels)
            loss_texts = crossentropyloss(logits_per_text, labels)
            final_loss = (loss_imgs + loss_texts) / 2.0

        final_loss.backward()
        optimizer.step()

        if opt.lr_sheduler != 'monitor':
            if id==0 or (id + 1) % 20 == 0:
                lr_scheduler.step(epoch + id / len(train_loader))

        epoch_loss += final_loss.item()
        period_loss += final_loss.item()
        period_cnt += 1

        if ((not opt.is_distributed) or (dist.get_rank() == 0)) and (id + 1) % opt.print_freq == 0:
            progress = (id + 1) / epoch_iters
            if (period_loss / period_cnt) < lowest_loss:
                lowest_loss = period_loss / period_cnt
                ## save models
                saving_models(opt.modeldir, "best.pth", epoch, model_image, fusion_model, model_text, optimizer)
            logging.info('Train Epoch: %d  | state: %d%%  |  loss: %.6f |  now_best: %.6f |  lr: %.8f  |  period_time: %.6f' % (epoch, int( 100 * progress), period_loss / period_cnt, lowest_loss, optimizer.param_groups[0]['lr'], time.time() - period_time))
            period_loss = 0
            period_cnt = 0
            period_time = time.time()

            saving_models(opt.modeldir, "lastest.pth", epoch, model_image, fusion_model, model_text, optimizer)

    if (not opt.is_distributed) or (dist.get_rank() == 0):    
        logging.info("="*114)
        logging.info(" " * 20 + "EPOCH:{} LOSS:{}".format(epoch, epoch_loss / epoch_iters))
        logging.info("="*114)