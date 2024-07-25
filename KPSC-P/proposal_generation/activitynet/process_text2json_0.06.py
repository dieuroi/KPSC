
import numpy as np
import json
import csv
import os, sys
import pandas as pd

anno_path = '/SSD2T/Datasets/ActivityNet/annotations/list_based/train.json'

duration_dict = {}
seg = 0.06
seg_list = [i for i in np.arange(seg, 1.0, seg)]
print(seg_list)

def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

data = load_json(anno_path)

for item in data:
  duration_dict[item["vid"]] = item["duration"]

final_list = []
for key in duration_dict.keys():
  dur = duration_dict[key]
  for posal in seg_list:
    for i in np.arange(0.0, 1.0, posal / 2.0):
      begin = i
      end = min(i + posal, 1.0)
      timestamp = [begin, end]
      sub_dict = {}
      sub_dict['timestamp'] = timestamp
      sub_dict['vid'] = key
      sub_dict['duration'] = dur
      final_list.append(sub_dict)
print(len(final_list))
with open('ActivityNet_train_proposal_generation_seg{}.json'.format(seg), 'w') as file_obj:
  json.dump(final_list, file_obj)