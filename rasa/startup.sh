#!/bin/sh

# ./usrun.sh -p RTX6000 --gpus=1 --time 48:0000 /netscratch/nehring/projects/modular-chatbot-dataset/frankenbot/research/testbed/code/lrec2022/rasa/startup.sh

export TORCH_HOME=/netscratch/nehring/cache/torch
export PIP_CACHE_DIR=/netscratch/nehring/cache/pip
pip install rasa==2.8.15 --ignore-installed ruamel.yaml
pip install requests
cd /netscratch/nehring/projects/modular-chatbot-dataset/frankenbot/research/testbed/code
python -m lrec2022.rasa.generate_training_data
lrec2022/rasa/train_models 
