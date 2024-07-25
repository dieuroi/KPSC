import json
def load_json(file_path, verbose=False):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

cha = load_json('charades_train_pseudo_supervision_TEP_PS.json')

noun_dict = {}

for item in cha:
    tokens = item['tokens']
    for i in tokens:
        if i in noun_dict.keys():
            noun_dict[i] += 1
        else:
            noun_dict[i] = 1

            
print(len(noun_dict.keys()))
