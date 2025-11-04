#!/bin/bash

pip install -r /home/aslam/TCAV-refact/requirements.txt
cd /home/aslam/TCAV-refact/src/captum
pip install .
cd /home/aslam/TCAV-refact/src
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
  --classifier default \
  --train_activation_layers avgpool \
  --test_activation_layers avgpool \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.001 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/bechmarking-ortho/with-concept-loss \
  --num_workers 8 \
  --correction_frequency 2 