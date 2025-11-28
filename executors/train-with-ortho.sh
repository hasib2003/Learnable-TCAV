#!/bin/bash

pip install -r /home/aslam/TCAV/requirements.txt
# pip install captum
pip install /home/aslam/TCAV/src/captum
# pip install .
cd /home/aslam/TCAV/src
# 

clear
echo ... requirements installed successfully ...
echo ... starting script ...

python -m pipelines.train-with-orthogonality \
  --train_dir /netscratch/aslam/TCAV/PetImages/train/with-cat-text \
  --test_dir  /netscratch/aslam/TCAV/PetImages/test \
  --concepts_dir /netscratch/aslam/TCAV/PetImages/Concepts/ \
  --model resnet18 \
  --concept_config_train config/ortho/config.json  \
  --concept_config_test  config/concept_test.json \
  --classifier signal \
  --train_activation_layers avgpool \
  --test_activation_layers avgpool \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.001 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/bechmarking-base/with-pretrained \
  --num_workers 8 \
  --pretrained \
  --correction_frequency 1 \
  --resume_checkpoint /netscratch/aslam/TCAV/text-inflation/bechmarking-base/with-pretrained/20251104_130214/best_model.pth