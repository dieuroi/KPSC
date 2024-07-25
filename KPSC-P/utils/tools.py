# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import numpy

def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)#[batchsize, 512]
    x2 = x2 / x2.norm(dim=-1, keepdim=True)#[batchsize, 512]

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def create_multiverbs_logits(images, texts, logit_scale):
    '''
    images:[batchsize, 512]
    texts:[batchsize, n_verbs, 512]
    '''
    b, n_verbs, dim = texts.shape
    texts = texts / texts.norm(dim=-1, keepdim=True)#[batchsize, n_verbs, 512]
    images = images / images.norm(dim=-1, keepdim=True)#[batchsize, 512]

    # cosine similarity as logits
    texts = texts.reshape(b * n_verbs, dim) #[batchsize*n_verbs, 512]
    logits_texts = logit_scale * texts @ images.t() #[batchsize*n_verbs, batchsize]
    logits_texts = logits_texts.reshape(b, n_verbs, b) #[batchsize, n_verbs, batchsize]

    logits_images = logit_scale * images @ texts.t() #[batchsize, batchsize*n_verbs]
    logits_images = logits_images.reshape(b, b, n_verbs).permute(0, 2, 1) #[batchsize, n_verbs, batchsize]

    logits_texts = logits_texts.max(dim = 1).values
    logits_images = logits_images.max(dim = 1).values

    # shape = [global_batch_size, global_batch_size]
    return logits_texts, logits_images


if __name__ == "__main__":
    import torch
    x1 = torch.rand([8, 512]).cuda()
    x2 = torch.rand([8, 6, 512]).cuda()
    logit_scale = torch.Tensor([100.]).cuda()
    logits_per_x1, logits_per_x2 = create_multiverbs_logits(x1, x2, logit_scale)
    print(logits_per_x1.shape, logits_per_x2.shape)