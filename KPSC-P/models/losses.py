import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torchvision.models as tmodels
from torchvision import models
import sys
sys.path.append('../')
from utils.tools import *

class contrast_loss(nn.Module):
    """contrast_loss."""
    def __init__(self, opt, logit_scale, ctx_dim):
        super(contrast_loss, self).__init__()
        self.logit_scale = logit_scale
        self.n_verb = opt.n_verb
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.batchsize = opt.batchsize
        self.ctx_dim = ctx_dim

    def forward(self, image_embedding, text_embedding, labels):
        if self.n_verb == 1:
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, self.logit_scale)
            loss_imgs = self.crossentropyloss(logits_per_image, labels)
            loss_texts = self.crossentropyloss(logits_per_text, labels)
            final_loss = (loss_imgs + loss_texts) / 2.0
        else:
            # select the hgihest verb logits as final logits
            text_embedding = text_embedding.reshape(self.batchsize, self.n_verb, self.ctx_dim)
            logits_per_image, logits_per_text = create_multiverbs_logits(image_embedding, text_embedding, self.logit_scale)
            loss_imgs = self.crossentropyloss(logits_per_image, labels)
            loss_texts = self.crossentropyloss(logits_per_text, labels)
            final_loss = (loss_imgs + loss_texts) / 2.0
        return final_loss