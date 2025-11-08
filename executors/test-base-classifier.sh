pip install -r requirements.txt

cd /home/aslam/TCAV/EXP-2/captum
pip install .

python /home/aslam/TCAV/EXP-1/src/test-classifier.py \
  --dataset_dir /netscratch/aslam/TCAV/PetImages/test \
  --checkpoint_path /netscratch/aslam/TCAV/text-inflation/EXP1/with-text/with-concept-loss/20251022_155007/last_model.pth \
  --num_classes 2 


