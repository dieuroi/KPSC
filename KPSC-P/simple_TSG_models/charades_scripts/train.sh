python ../train.py --model CrossModalityTwostageAttention \
--config configs/cha_simple_model/our_generated/simplemodel_cha_BS256_two-stage_attention.yml \
--seed 2022 \
--load_pretrained \
--load_model_path pretrained_best.pth >> train.log \