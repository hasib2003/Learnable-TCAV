#!/bin/bash

pip install -r /home/aslam/TCAV-refact/requirements.txt
cd /home/aslam/TCAV-refact/src/captum
pip install .
cd /home/aslam/TCAV-refact/src


clear
echo ... requirements installed successfully ...
echo ... starting script ...

python -m pipelines.train-with-concepts \
  --train_dir /netscratch/aslam/TCAV/PetImages/train/with-cat-text \
  --test_dir  /netscratch/aslam/TCAV/PetImages/test \
  --concepts_dir /netscratch/aslam/TCAV/PetImages/Concepts/ \
  --model resnet18 \
  --concept_config_train config/concept_test.json  \
  --concept_config_test  config/concept_train.json \
  --classifier default \
  --train_activation_layers avgpool \
  --test_activation_layers layer4.1.relu avgpool \
  --batch_size 64 \
  --epochs 4 \
  --lr 0.0001 \
  --checkpoint_dir /netscratch/aslam/TCAV/text-inflation/recreation-test/with-concept-loss \
  --num_workers 8 \
  --correction_frequency 2