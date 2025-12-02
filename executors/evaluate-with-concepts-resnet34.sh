#!/bin/bash

pip install -r /home/aslam/TCAV/requirements.txt
pip install captum


clear
echo ... requirements installed successfully ...
echo ... starting script ...

cd /home/aslam/TCAV/src

python -m pipelines.evaluate-concepts \
  --concepts_dir /netscratch/aslam/TCAV/dataset/textures \
  --random_prefix random500_\
  --model resnet34 \
  --mode exact \
  --concept_config /home/aslam/TCAV/src/config/imagenet/config2.json \
  --classifier signal \
  --target_layers layer2.3.relu layer3.3.relu layer3.5.relu layer4.0.relu layer4.1.relu layer4.2.relu \
  --save_dir /netscratch/aslam/Learnable-TCAV/fast-tcav/benchmarking/resnet34/