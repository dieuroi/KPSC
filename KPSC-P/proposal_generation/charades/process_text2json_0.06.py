
import numpy as np
import json
import csv
import os, sys
import pandas as pd

txt_path = '/SSD2T/Datasets/Charades/anno/charades_sta_train.txt'
csv_path = '/SSD2T/Datasets/datasets/Charades/anno/Charades_v1_train.csv'

duration_dict = {}

data = pd.read_csv(csv_path)
seg = 0.06
seg_list = [i for i in np.arange(seg, 1.0, seg)]
print(seg_list)

id_list = np.array(data['id'])
dur_list = np.array(data['length'])

assert len(id_list) == len(dur_list)

for i in range(len(id_list)):
    if id_list[i] in duration_dict.keys():
        assert dur_list[i] == duration_dict[id_list[i]]
    else:
        duration_dict[id_list[i]] = dur_list[i]

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

with open('Charades_train_proposal_generation_seg{}.json'.format(seg), 'w') as file_obj:
  json.dump(final_list, file_obj)