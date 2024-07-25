import random
import json
import os, sys

def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

random_ratio = 1.0 / 4.0

ori_path = 'ActivityNet_train_proposal_generation_selected_propoal_downline0.3_upline0.4.json'
result_path = './random_sample/ActivityNet_train_proposal_generation_selected_propoal_downline0.3_upline0.4.json_random{}.json'.format(random_ratio)


ori_data = load_json(ori_path)
result_list = []

for data in ori_data:
    if random.randint(1, 100) <= random_ratio * 100:
        result_list.append(data)

with open(result_path, 'w') as file_obj:
    json.dump(result_list, file_obj)
