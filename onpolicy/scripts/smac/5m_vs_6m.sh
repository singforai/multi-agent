#!/bin/sh

 #!/bin/sh
env="smac"
map="5m_vs_6m"
algo="mast" 
num_env_steps=10000000

for seed in 433 244 324 678 563;
do
    CUDA_VISIBLE_DEVICES=0 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_5m_vs_6m"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name mast_0015 --save_model --ppo_epoch 15 --clip_param 0.01
done