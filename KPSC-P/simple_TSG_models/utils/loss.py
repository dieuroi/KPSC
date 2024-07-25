import numpy as np
import torch
from torch import nn
import torch

class TAGLoss(nn.Module):
    def __init__(self):
        super(TAGLoss, self).__init__()

    def forward(self, w, mask):
        ac_loss = (-mask*torch.log(w+1e-8)).sum(1) / mask.sum(1)
        ac_loss = ac_loss.mean(0)

        return ac_loss

class TGRegressionCriterion(nn.Module):
    def __init__(self):
        super(TGRegressionCriterion, self).__init__()

        self.regloss1 = nn.SmoothL1Loss()
        self.regloss2 = nn.SmoothL1Loss()

    def forward(self, loc, s_gt, e_gt):

        total_loss = self.regloss1(loc[:,0], s_gt) + self.regloss2(loc[:,1], e_gt)

        return total_loss

class NLVLLoss(nn.Module):
    def __init__(self,cfg, reg_w=1):
        super().__init__()
        self.temporal_localization_loss = TGRegressionCriterion()
        self.temporal_attention_loss2 = TAGLoss()
        self.reg_w = reg_w

    def forward(self,model_outputs,batch):
        # position loss
        timestamps = model_outputs['timestamps'] # [B,2]
        gt_start_pos = batch["grounding_start_pos"]
        gt_end_pos = batch["grounding_end_pos"]
        gt_timestamps = torch.cat([gt_start_pos.unsqueeze(1),gt_end_pos.unsqueeze(1)],dim=1) # [B,2]

        localization_loss = self.temporal_localization_loss(timestamps, gt_start_pos, gt_end_pos)
        localization_loss = localization_loss * self.reg_w

        # attention loss
        attention_weights = model_outputs['attention_weights']  # [B,128]
        attention_masks = batch["attention_masks"]              # [B,128] 
        attention_loss = self.temporal_attention_loss2(attention_weights,attention_masks)
        
        loss_dict = {
            "localization_loss": localization_loss,
            "attention_loss": attention_loss
        }
        return loss_dict


class NLVLLoss_for_nverbs(nn.Module):
    def __init__(self,cfg, reg_w=1, method='min'):
        super().__init__()
        self.temporal_localization_loss = TGRegressionCriterion()
        self.temporal_attention_loss2 = TAGLoss()
        self.reg_w = reg_w

    def forward(self, model_outputs, batch):
        # position loss
        timestamps_list = model_outputs['timestamps'] # [B,2]
        gt_start_pos = batch["grounding_start_pos"]
        gt_end_pos = batch["grounding_end_pos"]
        gt_timestamps = torch.cat([gt_start_pos.unsqueeze(1),gt_end_pos.unsqueeze(1)],dim=1) # [B,2]
        min_localization_loss = 1e10
        for timestamps in timestamps_list:
            localization_loss = self.temporal_localization_loss(timestamps, gt_start_pos, gt_end_pos)
            localization_loss = localization_loss * self.reg_w
            min_localization_loss = min(min_localization_loss, localization_loss)

        # attention loss
        attention_weights_list = model_outputs['attention_weights']  # [B,128]
        attention_masks = batch["attention_masks"]              # [B,128] 
        min_attention_loss = 1e10
        for attention_weights in attention_weights_list:
            attention_loss = self.temporal_attention_loss2(attention_weights,attention_masks)
            min_attention_loss = min(min_attention_loss, attention_loss)
        
        loss_dict = {
            "localization_loss": min_localization_loss,
            "attention_loss": min_attention_loss
        }
        return loss_dict



class NLVLLoss_for_nverbs_basisloss(nn.Module):
    def __init__(self,cfg, reg_w=1, basis_w=1, method='min'):
        super().__init__()
        self.temporal_localization_loss = TGRegressionCriterion()
        self.temporal_attention_loss2 = TAGLoss()
        self.reg_w = reg_w

    def forward(self, model_outputs, batch):
        # position loss
        timestamps_list = model_outputs['timestamps'] # [B,2]
        gt_start_pos = batch["grounding_start_pos"]
        gt_end_pos = batch["grounding_end_pos"]
        gt_timestamps = torch.cat([gt_start_pos.unsqueeze(1),gt_end_pos.unsqueeze(1)],dim=1) # [B,2]
        
        min_localization_loss = 1e10
        for timestamps in timestamps_list:
            localization_loss = self.temporal_localization_loss(timestamps, gt_start_pos, gt_end_pos)
            localization_loss = localization_loss * self.reg_w
            min_localization_loss = min(min_localization_loss, localization_loss)

        # attention loss
        attention_weights_list = model_outputs['attention_weights']  # [B,128]
        attention_masks = batch["attention_masks"]              # [B,128] 
        min_attention_loss = 1e10
        for attention_weights in attention_weights_list:
            attention_loss = self.temporal_attention_loss2(attention_weights,attention_masks)
            min_attention_loss = min(min_attention_loss, attention_loss)
        

        # ctx_basis_loss    ctx[n_verb, n_ctx, ctx_dim]
        n_verb, n_ctx, dim = ctx.shape
        B = ctx.view(n_verb, -1)

        # compute orthogonalities btwn all baisis 
        D = torch.mm(B, torch.t(B)) 
        # make diagonal zeros
        D = (D - torch.eye(n_verb, n_verb, device="cuda"))**2
        basis_loss = torch.sum(D[0:n_verb,0:n_verb]) / (n_verb ** 2)
        basis_loss *= basis_w


        loss_dict = {
            "localization_loss": min_localization_loss,
            "attention_loss": min_attention_loss,
            "basis_loss": basis_loss
        }
        return loss_dict