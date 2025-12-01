#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <Job Name> <FilePath> [no of gpus] [Memory in GB] "
    echo "Example: $0 RTXA6000 train.sh 1 300"
    exit 1
fi

JOB_NAME="$1"
FILE_PATH="$2"
GPUS="${3:-0}"
MEMORY="${4:-256}"  # Default to 32GB if not provided

srun -K --partition="RTXA6000,RTXA6000-SDS,RTX3090,H100,H200,H200-SDS,RTX3090,batch,A100-40GB,A100-80GB" \
  --job-name="$JOB_NAME" \
  --gpus=${GPUS} \
  --cpus-per-task=8 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  --mem="${MEMORY}GB" \
  --container-mounts=/netscratch:/netscratch,/home:/home,/ds:/ds,"$(pwd)":"$(pwd)" \
  /bin/bash -c "chmod +x $FILE_PATH && ./$FILE_PATH"