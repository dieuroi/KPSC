import numpy as np
import torch
from torch import nn
import torch

def compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    if union == 0.0:
        if intersection > 0.0:
            return 1.0
        else:
            return 0.0
    return float(intersection) / union

def renew_best_score(cur, best, model):
    if best > cur:
        return best
    torch.save(model.state_dict(), 'pretrained_best.pth')

    return cur

class NLVLEvaluator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.iou_thresh = self.cfg.TRAIN.IOU_THRESH
        self.metrics = ["mIoU"] + ["Recall@{:.1f}".format(x) for x in self.iou_thresh]

    def __call__(self,model_outputs,batch):
        gt = torch.cat([batch["grounding_start_pos"].unsqueeze(1),batch["grounding_end_pos"].unsqueeze(1)],dim=1)
        pred = model_outputs['timestamps']
        scores = {x:[] for x in self.metrics}
        for p,g in zip(pred,gt):
            tiou = compute_tiou(p,g)
            scores["mIoU"].append(tiou)
            for thresh in self.iou_thresh:
                scores["Recall@{:.1f}".format(thresh)].append(tiou >= thresh)
        for k,v in scores.items():
            scores[k] = torch.mean(torch.Tensor(v))
        return scores



# class NLVLEvaluator_for_nverbs():
#     def __init__(self,cfg):
#         self.cfg = cfg
#         self.iou_thresh = self.cfg.TRAIN.IOU_THRESH
#         self.metrics = ["mIoU"] + ["Recall@{:.1f}".format(x) for x in self.iou_thresh]

#     def __call__(self, model_outputs, batch):
#         gt = torch.cat([batch["grounding_start_pos"].unsqueeze(1),batch["grounding_end_pos"].unsqueeze(1)],dim=1)
#         final_score = {}
#         max_mIou = 0.0
#         max_R7 = 0.0
#         for pred in model_outputs['timestamps']:
#             scores = {x:[] for x in self.metrics}
#             for p,g in zip(pred, gt):
#                 tiou = compute_tiou(p,g)
#                 scores["mIoU"].append(tiou)
#                 for thresh in self.iou_thresh:
#                     scores["Recall@{:.1f}".format(thresh)].append(tiou >= thresh)
#             for k, v in scores.items():
#                 scores[k] = torch.mean(torch.Tensor(v))

#             if scores['Recall@0.7'] > max_R7:
#                 max_R7 = scores['Recall@0.7']
#                 final_score = scores
#             elif scores['Recall@0.7'] == max_R7:
#                 if scores['mIoU'] > max_mIou:
#                     max_mIou = scores['mIoU']
#                     final_score = scores
#                 else:
#                     pass
#             else:
#                 pass
#         return final_score


class NLVLEvaluator_for_nverbs():
    def __init__(self,cfg):
        self.cfg = cfg
        self.iou_thresh = self.cfg.TRAIN.IOU_THRESH
        self.metrics = ["mIoU"] + ["Recall@{:.1f}".format(x) for x in self.iou_thresh]

    def __call__(self, model_outputs, batch):
        gt = torch.cat([batch["grounding_start_pos"].unsqueeze(1),batch["grounding_end_pos"].unsqueeze(1)],dim=1)
        final_score = {}
        max_mIou = 0.0
        max_R7 = 0.0
        cnt = 0
        for pred in model_outputs['timestamps']:
            scores = {x:[] for x in self.metrics}
            for p,g in zip(pred, gt):
                tiou = compute_tiou(p,g)
                scores["mIoU"].append(tiou)
                for thresh in self.iou_thresh:
                    scores["Recall@{:.1f}".format(thresh)].append(tiou >= thresh)
            for k, v in scores.items():
                scores[k] = torch.mean(torch.Tensor(v))

            if cnt == 0:
                final_score = scores
                max_R7 = scores['Recall@0.7']
                max_mIou = scores['mIoU']
            else:
                if scores['Recall@0.7'] > max_R7:
                    max_R7 = scores['Recall@0.7']
                    final_score = scores
                elif scores['Recall@0.7'] == max_R7:
                    if scores['mIoU'] > max_mIou:
                        max_mIou = scores['mIoU']
                        final_score = scores
                    else:
                        pass
                else:
                    pass
            cnt += 1
        return final_score