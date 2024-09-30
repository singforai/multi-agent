#!/bin/sh

env="smac"
map="MMM"
algo="mast" 
num_env_steps=1000000

for seed in 433 244 324 678 563;
do
    CUDA_VISIBLE_DEVICES=1 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${algo}\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name mast_02 --save_model --ppo_epoch 15 --clip_param 0.2
done

# env="smac"
# map="MMM"
# algo="mat" 
# num_env_steps=2000000

# for seed in 42 84 126 168 210;
# do
#     CUDA_VISIBLE_DEVICES=2 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${algo}\
#     --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --use_wandb --group_name ${algo} --save_model --layer_N 2
# done 

# env="smac"
# map="MMM"
# algo="rmappo" 
# num_env_steps=2000000

# for seed in 42 84 126 168 210;
# do
#     CUDA_VISIBLE_DEVICES=3 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${algo}\
#     --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --use_wandb --group_name ${algo} --save_model --use_feature_normalization
# done