
import torch.utils.data as data
import os
import os.path
import sys
sys.path.append('../')
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import json
import cv2
import numbers
import math
import torch
from utils.base_utils import *
import clip_code
from clip_code import clip


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, proposal_dict, datasets_root_path):
        self.proposal_dict = proposal_dict
        self.datasets_root_path = datasets_root_path

        self.path = os.path.join(self.datasets_root_path, self.proposal_dict['vid'])
        self.duration = self.proposal_dict['duration']
        self.vid = self.proposal_dict['vid']
        self.timestamp = self.proposal_dict['timestamp']
        self.tokens = self.proposal_dict['tokens']
        self.all_video_frame_num = len(os.listdir(self.path))
        self.num_frames = self.get_num_frames()


    def get_num_frames(self):
        timestamp = self.timestamp
        assert timestamp[1] >= timestamp[0]
        num_frames = int(self.all_video_frame_num * (timestamp[1] - timestamp[0]))
        return num_frames



class DATASETS(data.Dataset):
    def __init__(self, json_file, datasets_root_path,
                 num_segments=8, new_length=1,
                 image_tmpl='{}-{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.json_file = json_file
        self.datasets_root_path = datasets_root_path
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias

        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx, video_name):
        image_path = os.path.join(directory, self.image_tmpl.format(video_name, idx))
        #print("loading from {}".format(image_path))
        return [Image.open(image_path).convert('RGB')]
    
    def _parse_list(self):
        datasets_proposal_list = load_json(self.json_file)
        self.video_list = [VideoRecord(x, self.datasets_root_path) for x in datasets_proposal_list]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames,
                        size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        #print("duration:", record.duration, "timestamp:", record.timestamp, "num_frames:", record.num_frames, "vid:", record.vid, "all_video_frame_num:", record.all_video_frame_num)
        ## get begin shift
        begin_shift = int(record.timestamp[0] * record.all_video_frame_num)
        segment_indices = segment_indices + begin_shift
        #print(segment_indices)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
    @property
    def total_length(self):
        return self.num_segments * self.seg_length


    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p, record.vid)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)

        process_data = self.transform(images)

        ## process token:
        mess = " ".join(record.tokens) + '.'
        token = clip.tokenize(mess).squeeze(0) #[L]

        return process_data, token, len(record.tokens)

    def __len__(self):
        return len(self.video_list)


if __name__ == "__main__":
    # image_tmpl = '{}-{:06d}.jpg'
    # image_path = os.path.join(os.path.join("/home/yushui/mc2/datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb", '0A8CF'), image_tmpl.format('0A8CF', 4))
    # image = [Image.open(image_path).convert('RGB')]
    # print(type(image[0]))

    # charades = load_json('/home/yushui/mc2/wuxun/wuxun/temporal_sentence_grounding/zero-shot-TSG/clip_based_zero_shot_TSG/psvl_anno/charades/charades_train_pseudo_supervision_TEP_PS.json')
    # print(type(charades))
    # print(type(charades[0]))

    # t = sorted(os.listdir('/home/yushui/mc2/datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb/0A8CF'))
    # print(t)
    datasets = DATASETS('/home/yushui/mc2/wuxun/wuxun/temporal_sentence_grounding/zero-shot-TSG/clip_based_zero_shot_TSG/psvl_anno/charades/charades_train_pseudo_supervision_TEP_PS.json',
                        '/home/yushui/mc2/datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb',
                 num_segments=8, new_length=1,
                 image_tmpl='{}-{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1)

    print(datasets.__getitem__(0))
