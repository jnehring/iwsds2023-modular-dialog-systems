# Module Selection: A New Task and a New Benchmark Dataset for DialogSystems

## Introduction
Dialog systems often comprise Natural Language Understanding and a rule-based dialog manager, an architecture we call Intent-Based Dialog System. In practical applications, dialog systems often combine multiple Intent-Based Dialog Systems. Although this combination is not trivial, this topic has, to our best knowledge, not been addressed by researchers yet. Therefore we introduce a new task Module Selection, which is the problem of selecting the correct sub-dialog system for a given user utterance. Further, we provide a dataset to evaluate models for Module Selection. We propose three models for Module Selection and evaluate them using the proposed dataset and evaluation methodology. The models serve as a strong baseline for future research in Module Selection.

You can find the dataset we published in the final_data folder in this repository.

## Dataset Creation

### split_data.py
Read the input file and create the splitted dataset.

to run it:
```
python -m split_data
```

## Get responses and merge all responses for modular chatbot experiments

### google_dialogflow/get_responses.py
Create google dialogflow chatbots for each splits. Then precess it by each google dialogflow chatbots and save the responses.

to run it:
```
python -m google_dialogflow.get_responses
```

### google_dialogflow/merge_dialogflow.py
Merge all split responses for each google dialogflow chatbot. Then save all the responses.

to run it:
```
python -m google_dialogflow.merge_dialogflow
```

### ibm_watson_assistant/get_responses.py
Create ibm watson assistant chatbots for each splits. Then precess it by each ibm watson assistant chatbots and save the responses.

to run it:
```
python -m ibm_watson_assistant.get_responses
```

### ibm_watson_assistant/merge_ibm_watson.py
Merge all split responses for each ibm watson assistant chatbot. Then save all the responses.

to run it:
```
python -m ibm_watson_assistant.merge_ibm_watson
```

### rasa/generate_training_data.py
Create training data for rasa chatbots.

to run it:
```
python -m rasa.generate_training_data
```

### rasa/README.md
This file have information about how to create rasa chatbot for each split.
To do that go to each split directory. Then do the followings for each split.

This code snippet is an example for inhomogenous rasa split 0:
```
cd rasa/tempfolder/rasa/0
CUDA_VISIBLE_DEVICES=0 
rasa train nlu --config ../../../config.yml
```

### rasa/process_data.py
Create rasa chatbots for each splits. Then precess it by each rasa chatbots and save the responses.

to run it:
```
python -m rasa.process_data
```

### rasa/merge_rasa.py
Merge all split responses for each rasa chatbot. Then save all the responses.

to run it:
```
python -m rasa.merge_rasa
```

### merge_all_datasets.py
Merge all chatbot data and create homogenous and inhomogenous dataset for modular dialogsystem.

to run it:
```
python -m merge_all_datasets
```

## Models and Experiments: Script Overview (Need an update)

### LREC_2022_split_0.ipynb
Read the modular_dialogsystem_dataset.csv file. Then perform all the three experiments and calculate final results for split 0.

### LREC_2022_split_1.ipynb
Read the modular_dialogsystem_dataset.csv file. Then perform all the three experiments and calculate final results for split 1.

### LREC_2022_split_2.ipynb
Read the modular_dialogsystem_dataset.csv file. Then perform all the three experiments and calculate final results for split 2.

### LREC_2022_split_3.ipynb
Read the modular_dialogsystem_dataset.csv file. Then perform all the three experiments and calculate final results for split 3.

### LREC_2022_split_4.ipynb
Read the modular_dialogsystem_dataset.csv file. Then perform all the three experiments and calculate final results for split 4.

