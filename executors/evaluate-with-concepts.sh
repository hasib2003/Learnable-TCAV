#!/bin/bash

pip install -r /home/aslam/TCAV/requirements.txt
pip install captum


clear
echo ... requirements installed successfully ...
echo ... starting script ...

cd /home/aslam/TCAV/src

# /netscratch/aslam/TCAV/dataset/textures
python -m pipelines.evaluate-concepts \
  --concepts_dir /netscratch/aslam/TCAV/dataset/textures \
  --random_prefix random500_\
  --num_classes 2\
  --model resnet18 \
  --mode all \
  --concept_config config/low_level_concepts.json \
  --classifier signal \
  --target_layers layer2.1.relu layer3.1.relu layer4.1.relu avgpool\
  --save_dir /netscratch/aslam/TCAV/fast-tcav/benchmarking/cat-dog/ \
  --checkpoint /netscratch/aslam/TCAV/text-inflation/bechmarking-base/with-pretrained/20251104_130214/best_model.pth