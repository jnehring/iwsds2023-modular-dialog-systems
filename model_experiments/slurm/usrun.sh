#!/bin/sh
# ./usrun.sh -p RTXA6000 --gpus=1 --mem 80000 --cpus-per-gpu=1 --pty --time 08:00:00 /bin/bash

# start the dialogpt enroot container on the cluster
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.12-py3.sqsh

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$HOME \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  $*
