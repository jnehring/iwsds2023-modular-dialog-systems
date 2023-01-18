# Module Selection: A new task for Dialog Systems

## Introduction

Many different dialog systems exist, which usually cover limited domains. This paper examines the Modular Dialog System Framework to combine many conversational agents to create a unified, diverse dialog system. The Modular Dialog System treats the underlying conversational agents as black boxes and works with any dialog system without further adaption. It also works with commercial frameworks, such as Google Dialogflow or IBM Watson Assistant, in which the inner workings are unknown company secrets. We propose a new task, Module Selection, choosing a conversational agent for a user utterance. Also, we propose an evaluation methodology for Modular Dialog Systems. Using the three available commercial frameworks, Google Dialogflow, Rasa, and IBM Watson Assistant, we create a dataset and propose three models that serve as a strong baseline for future research in Module Selection. Also, we examine the performance difference between a Modular Dialog System and the same dialog system implement in a single, monolithic system. We publish our dataset and source codes as open source.

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

