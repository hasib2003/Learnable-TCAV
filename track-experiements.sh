
echo execute from outside fastcav dir

pip install ./fastcav

cd fastcav

echo "increasing the limit of open files to 65535"

ulimit -n 65535


python experiments/track_concepts_during_training.py \
    dataset=imagenet \
    checkpoint=timm/resnet50 \
    workers=8 \
    concept_set=textures \
    "modules=[layer1,layer2,layer3,layer4]" \
    trainer.max_epochs=90 \
    loss=cross_entropy \
    metric=multiclass \
    batch_size=256 \
    device=cuda \
    data_dir=/netscratch/aslam/TCAV/dataset\
    dataset.dataset.train.root=/ds/images/imagenet/ \
    dataset.dataset.test.root=/ds/images/imagenet/val_folders
