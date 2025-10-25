srun -K -p H200,H100,H200-SDS,RTXA6000,RTXA6000-SDS,A100-80GB,A100-40GB,A100-IML \
    --ntasks=1 \
    --gpus-per-task=1\
    --cpus-per-task=16 \
    --mem=512GB \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh  \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/fscratch:/fscratch,/home/aslam:/home/aslam,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`"  \
    /bin/bash -c "chmod +x track-experiements.sh && ./track-experiements.sh"
