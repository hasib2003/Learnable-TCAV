#!/bin/bash

pip install -r /home/mukhtar/Learnable-TCAV/requirements.txt
pip install captum


clear
echo ... requirements installed successfully ...
echo ... starting script ...

cd /home/mukhtar/Learnable-TCAV/src

# /netscratch/mukhtar/Learnable-TCAV/dataset/textures
python -m pipelines.evaluate-concepts \
  --concepts_dir /netscratch/mukhtar/Learnable-TCAV/broden1_224/textures \
  --random_prefix random500_\
  --num_classes 2 \
  --model resnet18 \
  --mode exact \
  --concept_config /netscratch/mukhtar/Learnable-TCAV/src/config/imagenet/config2.json \
  --classifier signal \
  --target_layers layer2.1.relu layer3.1.relu layer4.1.relu \
  --save_dir /netscratch/mukhtar/Learnable-TCAV/fast-tcav/benchmarking/cat-dog/ \
  --checkpoint /netscratch/mukhtar/Learnable-TCAV/text-inflation/bechmarking-base/with-pretrained/20251104_130214/best_model.pth