#!/bin/sh

# start a slurm container
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.02-py3.sqsh
srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$HOME \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  $*

