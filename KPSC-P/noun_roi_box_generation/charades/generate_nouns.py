import os, sys
sys.path.append('../mmdetection')
from mmdet.apis import init_detector, inference_detector
import mmcv
import json
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps

class VideoRecord(object):
    def __init__(self, proposal_dict, datasets_root_path):
        self.proposal_dict = proposal_dict
        self.datasets_root_path = datasets_root_path

        self.path = os.path.join(self.datasets_root_path, self.proposal_dict['vid'])
        self.duration = self.proposal_dict['duration']
        self.vid = self.proposal_dict['vid']
        self.timestamp = self.proposal_dict['timestamp']
        self.all_video_frame_num = len(os.listdir(self.path))
        self.num_frames = self.get_num_frames()
        self.bis = self.timestamp[1] - self.timestamp[0]


    def get_num_frames(self):
        timestamp = self.timestamp
        assert timestamp[1] >= timestamp[0]
        num_frames = int(self.all_video_frame_num * (timestamp[1] - timestamp[0]))
        return num_frames

def _get_val_indices(record):
    total_length = 8
    index_bias = 1
    num_segments = 8
    seg_length = 1
    if record.num_frames <= total_length:
        return np.array([i * record.num_frames // total_length
                                             for i in range(total_length)], dtype=np.int32) + index_bias
    offset = (record.num_frames / num_segments - seg_length) / 2.0
    return np.array([i * record.num_frames / num_segments + offset + j
                        for i in range(num_segments)
                        for j in range(seg_length)], dtype=np.int32) + index_bias

def get_path_list(record, indices):
    image_tmpl = '{}-{:06d}.jpg'
    path_list = []
    for i, seg_ind in enumerate(indices):
        p = int(seg_ind)
        image_path = os.path.join(record.path, image_tmpl.format(record.vid, p))
        path_list.append(image_path)
    return path_list

def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

'''
['person' 'bicycle' 'car' 'motorcycle' 'airplane' 'bus' 'train' 'truck'
 'boat' 'traffic light' 'fire hydrant' 'stop sign' 'parking meter' 'bench'
 'bird' 'cat' 'dog' 'horse' 'sheep' 'cow' 'elephant' 'bear' 'zebra'
 'giraffe' 'backpack' 'umbrella' 'handbag' 'tie' 'suitcase' 'frisbee'
 'skis' 'snowboard' 'sports ball' 'kite' 'baseball bat' 'baseball glove'
 'skateboard' 'surfboard' 'tennis racket' 'bottle' 'wine glass' 'cup'
 'fork' 'knife' 'spoon' 'bowl' 'banana' 'apple' 'sandwich' 'orange'
 'broccoli' 'carrot' 'hot dog' 'pizza' 'donut' 'cake' 'chair' 'couch'
 'potted plant' 'bed' 'dining table' 'toilet' 'tv' 'laptop' 'mouse'
 'remote' 'keyboard' 'cell phone' 'microwave' 'oven' 'toaster' 'sink'
 'refrigerator' 'book' 'clock' 'vase' 'scissors' 'teddy bear' 'hair drier'
 'toothbrush']
'''

if __name__ == "__main__":
    # Specify the path to model config and checkpoint file
    config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'  
    checkpoint_file = '../mmdetection/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    score_thr = 0.3
    datasets_root_path = '/SSD2T/Datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb'

    json_list = ['../proposal_generation/charades/random_sample/Charades_train_proposal_generation_selected_propoal_downline0.2326_upline0.35_random0.6666666666666666.json']

    save_list = ['./generated_nouns/Charades_train_proposal_generation_selected_propoal_downline0.2326_upline0.35_random0.6666666666666666_tokens.json']
    
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda')
    CLASSES = np.array(model.CLASSES)
    for dinx in range(len(json_list)):
        json_path = json_list[dinx]
        save_path = save_list[dinx]
        
        all_final_data = []
        json_dict_list = load_json(json_path)
        for per_data in tqdm(json_dict_list):
            video = VideoRecord(per_data, datasets_root_path)
            segment_indices = _get_val_indices(video)
            begin_shift = int(video.timestamp[0] * video.all_video_frame_num)
            segment_indices = segment_indices + begin_shift
            path_list = get_path_list(video, segment_indices)

            scores_dict = {}
            for img in path_list:
                # test a single image and show the results
                result = inference_detector(model, img)
                ## results ananlysis
                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = result, None
                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)

                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > score_thr
                bboxes = bboxes[inds, :]
                selected_scores = bboxes[:, -1]
                labels = labels[inds]
                assert len(labels) == len(selected_scores)

                for i in range(len(labels)):
                    if labels[i] in scores_dict.keys():
                        scores_dict[labels[i]] = max(scores_dict[labels[i]], selected_scores[i])
                    else:
                        scores_dict[labels[i]] = selected_scores[i]

            # sort and select top5 nouns
            sorted_score = sorted(scores_dict.items(),key=lambda x:x[1], reverse=True)
            if len(sorted_score) < 5:
                continue
            else:
                sorted_score = sorted_score[:5]
                labels_index = [x[0] for x in sorted_score]
                labels_names = CLASSES[labels_index]
                per_data['tokens'] = labels_names.tolist()
                all_final_data.append(per_data)
            #print(selected_scores)

        ## save results
        with open(save_path, 'w') as file_obj:
            json.dump(all_final_data, file_obj)