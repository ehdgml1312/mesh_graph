#!/bin/sh
export CUDA_VISIBLE_DEVICES=2

python3 train.py -c config/edge.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3

python3 train.py -c config/trans.yaml &
sleep 3



