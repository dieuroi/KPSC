## Pseudo Supervision Construction

```
cd ../KPSC-F
```

### Step 1. Pseudo Captions And Extract Features

We use the pre-extracted BLIP captions and features at [SPL](https://github.com/minghangz/SPL), if you want to generate and extract the features yourself, follow the first and second steps from SPL.

Put the BLIP captions and features at **dataset/BLIP**

File directory structure of the dataset as shown:

```
dataset/
├── annotations
│   ├── activitynet
│   │   └── activitynet-SPL-train.json
│   └── charades
│       └── charades_train.json
├── BLIP
│   ├── activitynet
│   │   ├── blip_features
│   │   ├── dense_caption_features
│   │   └── dense_captions
│   └── charades
│       ├── blip_features
│       ├── dense_caption_features
│       └── dense_captions
└── Pseudo-dataset
    ├── activitynet
    └── charades
```



### Step 2. Generate Pseudo Lables

```
# ActivityNet-Captions
python KPSC-F-Pseudo-label-gen.py --task activitynet --top_k 25 --Allow_mask 30

# Charades-STA
python KPSC-F-Pseudo-label-gen.py --task activitynet --top_k 25 --Allow_mask 1
```



## Training

We use [EMB](https://github.com/Raymond-sci/EMB) as KPSC-F base model, please follow [EMB](https://github.com/Raymond-sci/EMB)'s instruction to download features and prepare python environment.

```
git clone https://github.com/Raymond-sci/EMB.git
cd EMB
you need to change the path of pseudo-training dataset in the line 45 and line 80 in util/data_gen.py, like

# for ActivityNet-Captions
#in the line 80, change 
train_data = load_json(os.path.join(data_dir, 'train.json'))
#to
train_data = load_json(os.path.join(data_dir, 'M_A30_activitynet_T25.json'))

# for Charades-STA
#in the line 45, change 
train_data = load_lines(os.path.join(data_dir, 'charades_sta_train.txt'))
#to
train_data = load_lines(os.path.join(data_dir, 'M_A1_charades_T25.txt'))
```



```
# ActivityNet-Captions
python main.py --task activitynet --char_dim 100 
```



## Evaluting

```
mkdir EMB/sessions
```

download the pre-trained model from [xxx], and put them at EMB/sessions, and move ours pre-generated pseudo-training datasets to EMB/dataset/dataset

```
#for ActivityNet-Captions
cp EMB/sessions/activitynet/M_A30_activitynet_T25/M_A30_activitynet_T25.json EMB/data/dataset/activitynet
#for Charades-STA
cp EMB/sessions/charades/M_A1_charades_T25/M_A1_charades_T25.txt EMB/data/dataset/charades
```

then you need to change the training dataset path in EMB/util/data_gen.py

```
# for ActivityNet-Captions
#in the line 80, change 
train_data = load_json(os.path.join(data_dir, 'train.json'))
#to
train_data = load_json(os.path.join(data_dir, 'M_A30_activitynet_T25.json'))


# for Charades-STA
#in the line 45, change 
train_data = load_lines(os.path.join(data_dir, 'charades_sta_train.txt'))
#to
train_data = load_lines(os.path.join(data_dir, 'M_A1_charades_T25.txt'))
```



```
cd EMB
# ActivityNet-Captions
python main.py --task activitynet --mode test --model_name M_A30_activitynet_T25

# Charades-STA
python main.py --task charades --mode test --model_name M_A1_charades_T25
```