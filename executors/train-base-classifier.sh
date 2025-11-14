#!/bin/bash

pip install -r /home/aslam/TCAV-refact/requirements.txt

cd /home/aslam/TCAV-refact/src

python -m pipelines.train-base-classifier \
  --train_dir /netscratch/aslam/TCAV/PetImages/train/with-cat-text \
  --test_dir /netscratch/aslam/TCAV/PetImages/test \
  --batch_size 64 \
  --epochs 4 \
  --lr 0.0001 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/EXP1/with-text/base \
  --num_workers 8  \
  --pretrained \
  --freeze_backbone 