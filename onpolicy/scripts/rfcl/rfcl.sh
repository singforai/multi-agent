#!/bin/bash

python ../../train_grf.py \
    --n_rollout_threads 20 \
    --n_eval_rollout_threads 20 \
    --eval_interval 15 \
    --eval_episodes 20 \
    --clip_param 0.05 \
    --ppo_epoch 10 \
    --episode_length 200 \
    --algorithm_name mast \
    --n_head 4 \
    --save_model \
    --save_interval 500 \
    --experiment_name demo1 \
    --num_gpu 1 \
    --group_name rfcl \ 