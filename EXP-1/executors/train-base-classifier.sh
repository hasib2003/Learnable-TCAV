#!/bin/bash

pip install -r requirements.txt
pip install captum

# cd /home/aslam/TCAV/EXP-2/captum
# pip install -e .[dev]

python /home/aslam/TCAV/EXP-1/src/train-base-classifier.py \
  --train_dir /netscratch/aslam/TCAV/PetImages/train/with-cat-text \
  --test_dir /netscratch/aslam/TCAV/PetImages/test \
  --batch_size 64 \
  --epochs 2 \
  --lr 0.0001 \
  --train_split 0.80 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/EXP1/with-text/no-concept-loss \
  --num_workers 8  