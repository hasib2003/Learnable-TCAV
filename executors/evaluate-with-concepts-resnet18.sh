#!/bin/bash

pip install -r /netscratch/mukhtar/Learnable-TCAV/requirements.txt
pip install captum


clear
echo ... requirements installed successfully ...
echo ... starting script ...

cd /netscratch/mukhtar/Learnable-TCAV/src

# /netscratch/mukhtar/Learnable-TCAV/dataset/textures
python -m pipelines.evaluate-concepts \
  --concepts_dir /netscratch/mukhtar/Learnable-TCAV/dtd/images \
  --random_prefix random500_\
  --model resnet18 \
  --mode exact \
  --concept_config /netscratch/mukhtar/Learnable-TCAV/src/config/imagenet/config2.json \
  --classifier signal \
  --target_layers layer1.0.relu layer1.1.relu layer2.0.relu layer2.1.relu layer3.0.relu layer3.1.relu layer4.0.relu layer4.1.relu \
  --save_dir /netscratch/mukhtar/Learnable-TCAV/fast-tcav/benchmarking/resnet18/
