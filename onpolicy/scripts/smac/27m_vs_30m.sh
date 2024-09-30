#!/bin/sh

env="smac"
map="27m_vs_30m"
algo="mast" 
num_env_steps=10000000

for seed in 433 244 324 678 563;
do
    CUDA_VISIBLE_DEVICES=3 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_27m_vs_30m"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name mast_025 --save_model --ppo_epoch 5 --clip_param 0.2
done