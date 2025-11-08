#!/bin/bash


pip install -r requirements.txt

pip install /home/aslam/TCAV/Captum-Cuda

python /home/aslam/TCAV/EXP-1/src/train-with-concepts.py \
  --train_dir /netscratch/aslam/TCAV/PetImages/train/with-cat-text \
  --test_dir /netscratch/aslam/TCAV/PetImages/test \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-5 \
  --train_split 0.80 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/EXP1/with-text/with-concept-loss/cuda-captum \
  --num_workers 8 \
  --c_corr_after 1 \
  --target_layer avgpool \
  --resume_checkpoint /netscratch/aslam/TCAV/text-inflation/EXP1/with-text/with-concept-loss/cuda-captum/20251028_174407/best_model.pth
