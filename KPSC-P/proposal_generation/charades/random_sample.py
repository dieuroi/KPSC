import random
import json
import os, sys

def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

random_ratio = 0.06

ori_path = 'Charades_train_proposal_generation_selected_propoal_line0.3.json'
result_path = './random_sample/Charades_train_proposal_generation_selected_propoal_line0.3_random{}.json'.format(random_ratio)


ori_data = load_json(ori_path)
result_list = []

for data in ori_data:
    if random.randint(1, 100) <= random_ratio * 100:
        result_list.append(data)

with open(result_path, 'w') as file_obj:
    json.dump(result_list, file_obj)
