#!/bin/bash

python ../../train_grf.py --use_render\
    --save_replay \
    --num_env_steps 100000000 \
    --model_dir ./models/demo1/mast/3600 \
    --algorithm_name mast \
    --n_head 4 \
    --num_gpu 0 \
    