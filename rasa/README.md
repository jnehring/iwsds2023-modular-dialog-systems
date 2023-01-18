# Here we create rasa nlu models and process the rasa part of the dataset with them

```
cd /netscratch/nehring/projects/modular-chatbot-dataset/frankenbot/research/testbed/code/lrec2022/rasa
./usrun.sh -p RTX6000 --gpus=1 --time 48:0000 --pty /bin/bash
```

# Install requirements, generate data
```
export TORCH_HOME=/netscratch/nehring/cache/torch
export PIP_CACHE_DIR=/netscratch/nehring/cache/pip
pip install rasa --ignore-installed ruamel.yaml
pip install requests
cd /netscratch/nehring/projects/modular-chatbot-dataset/frankenbot/research/testbed/code
python -m lrec2022.rasa.generate_training_data
```

# train all models
```
/rasa/train_models 
```