import os, sys
import torch
import json
from torch import nn
sys.path.append("../")
sys.path.append("../../../")
from models import *
from proposal_datasets import *
import clip_code
from datasets.transforms_ss import *
from RandAugment import RandAugment
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm

def get_augmentation():
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 224 * 256 // 224
    unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                GroupCenterCrop(224)])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])

def frozen(net):
    for name, parameter in net.named_parameters():
        parameter.requires_grad = False

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

device = torch.device("cuda")
anno_path = 'Charades_train_proposal_generation_seg0.06.json'
datasets_root_path = '/SSD2T/Datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb'
line = 0.55
print("line:", line)
batch_size = 128


transform_val = get_augmentation()
model, clip_state_dict = clip_code.load('../clip_code/checkpoints/ViT-B-32.pt', 
            device="cuda", jit=False, tsm=False, T=8, dropout=0.0 , emb_dropout=0.0 ,pretrain=True, joint = False) #Must set jit=False for training  ViT-B/32


anno_data = PROPOSAL_DATASETS(anno_path, datasets_root_path, 8, image_tmpl='{}-{:06d}.jpg', random_shift=False, transform=transform_val)

anno_loader = Data.DataLoader(dataset=anno_data, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=False,
                            drop_last=True)

print("making datasets done! proposal generation and selection...")

image_model1 = ImageCLIP(model)
image_model2 = ImageCLIP(model)
image_model1 = image_model1.to(device)
image_model2 = image_model2.to(device)
frozen(image_model1)
frozen(image_model2)

final_results = []

for id, (full_image_input, sub_img_input, bis, messes) in enumerate(tqdm(anno_loader)):
    sub_img_input = sub_img_input.to(device)
    full_image_input = full_image_input.to(device)

    sub_img_input = sub_img_input.view((-1, 8, 3) + sub_img_input.size()[-2:])#[32, 8, 3, 224, 224]
    b, t, c, h, w = sub_img_input.size()
    sub_img_input = sub_img_input.view(-1, c, h, w)

    full_image_input = full_image_input.view((-1, 8, 3) + full_image_input.size()[-2:])#[32, 8, 3, 224, 224]
    b, t, c, h, w = full_image_input.size()
    full_image_input = full_image_input.view(-1, c, h, w)

    assert b == batch_size

    t1 = image_model1(sub_img_input)
    t2 = image_model2(full_image_input)

    t1 = t1.reshape(b, t*512)
    t2 = t2.reshape(b, t*512)

    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = t2 / t2.norm(dim=-1, keepdim=True)
    #print(t1.shape, t2.shape) #[32, 8, 512]

    logits = t1 @ t2.t() # 32 *32
    for i in range(batch_size):
        log = logits[i][i].item() * bis[i]
        if log > line:
            sub_dict = {}
            sub_dict['vid'] = messes['vid'][i]
            sub_dict['timestamp'] = [messes['timestamp'][0][i].item(), messes['timestamp'][1][i].item()]
            sub_dict['duration'] = messes['duration'][i].item()
            final_results.append(sub_dict)

print("All results is {}".format(len(final_results)))

with open('Charades_train_proposal_generation_selected_propoal_line{}.json'.format(line), 'w') as file_obj:
    json.dump(final_results, file_obj)


