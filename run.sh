#!/bin/sh
export CUDA_VISIBLE_DEVICES=2

python3 train.py -c config/avg.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3

python3 train.py -c config/avg2.yaml &
sleep 3



