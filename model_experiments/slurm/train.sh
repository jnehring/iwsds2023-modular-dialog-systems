#!/bin/sh

# ./usrun.sh -p RTX6000 --gpus=1 --mem 80000 --cpus-per-gpu=1 --time 08:00:00 /netscratch/akahmed/modular-ds-dataset/model_experiments/slurm/train.sh model_experiments/datasets/modular_dialogsystem_dataset_homogenous_ibm_watson_assistant.csv full_experiment
# ./usrun.sh -p RTXA6000 --gpus=1 --mem 80000 --cpus-per-gpu=1 --pty --time 08:00:00 /bin/bash

export TORCH_HOME=/netscratch/akahmed/cache/torch
export PIP_CACHE_DIR=/netscratch/akahmed/cache/pip
export PIP_DOWNLOAD_DIR=/netscratch/akahmed/cache/pip

 # i get a weird error without the following line because somebody sets LOCAL_RANK=0 which confuses everything...
export LOCAL_RANK=-1 

cd /netscratch/akahmed/modular-ds-dataset
pip install -r model_experiments/requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

python -m model_experiments.train --input_file $1 --experiment $2
