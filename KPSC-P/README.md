## Training
### Pseudo Supervision Construction
#### Step1: Proposal Generation
Take Charades-STA for example, run the following scripts.

```
cd proposal_generation/charades
python process_text2json_0.06.py
python proposal_generation.py
python random_sample.py
```

After this, a json file will be stored in **random_sample**.

#### Step2: Noun Generation
1. Setup [mmdetection](https://github.com/open-mmlab/mmdetection) in **mmdetection**.
2. Take Charades-STA for example, run the following scripts.
   
```
cd ../noun_roi_box_generation/charades
python generate_nouns.py
```
After this, a json file will be stored in **generated_nouns**.

#### Step3: Train Verb prompts
Get back to KPSC-P, run the following scripts.

```
python train.py

```
After this, a json file will be stored in **train_results**.

### Train a simple TSG model based on pseudo supervision
Take Charades-STA for example, run the following scripts.

```
cd simple_TSG_models/charades_scripts
./train.sh
```

## Acknowledgments

Some of this code is borrowed from the following repositories:

* [ActionCLIP](https://github.com/sallymmx/ActionCLIP)
* [PSVL](https://github.com/gistvision/PSVL)