#!/bin/bash

pip install -r /home/aslam/TCAV/requirements.txt


clear
echo ... requirements installed successfully ...
echo ... starting script ...

cd /home/aslam/TCAV/src

# /netscratch/aslam/TCAV/dataset/textures
python -m pipelines.quantify-linearity \
  --data_dir /netscratch/aslam/TCAV/PetImages/test/   \
  --model resnet18 \
  --start_module avgpool\
  --save_dir /netscratch/aslam/TCAV/fast-tcav/ \
  --checkpoint /netscratch/aslam/TCAV/text-inflation/bechmarking-base/with-pretrained/20251110_092506/best_model.pth
