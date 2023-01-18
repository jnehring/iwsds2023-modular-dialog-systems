'''
to run it:
python -m model_experiments.train --input_file model_experiments/datasets/modular_dialogsystem_dataset_homogenous_google_dialogflow.csv --experiment single_experiment
'''
import numpy as np
import pandas as pd
import shutil
import os
import time
import argparse
import logging
import json
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report 
from model_experiments.util import init_experiments, read_args, write_results
from transformers import BertModelWithHeads, BertConfig, BertForSequenceClassification, EarlyStoppingCallback, TrainingArguments, Trainer, EvalPrediction, AdamW, BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from ast import literal_eval
from sklearn.utils.class_weight import compute_class_weight
from math import log2
from tqdm.auto import tqdm
from model_experiments.model_confidence import BERT_BaseArch_Confidence
from model_experiments.model_text import BERT_BaseArch_Text
from model_experiments.model_joined import BERT_BaseArch_Joint
from model_experiments.dataloader import load_modular_dataset
from model_experiments.mod_ic import ModularIntentClassification


device = torch.device("cuda")

# create a boxplot to see indomain and outdomain confidence values
def create_box_plot(args, log_dir):
    plt.clf()
    df = pd.read_csv(args.input_file)
    split = 0
    train_df = df[(df.dataset == "train_ms") & (df.split == split)]
    train_nlu_df = df[(df.dataset == "train_nlu") & (df.split == split)]
    train_df = pd.concat([train_df, train_nlu_df])

    # parse and return the confidence
    def get_confidence(t_row):
        ls = literal_eval(t_row)
        row = json.loads(str(ls))
        return row

    agent_0_in = []
    agent_0_out = []
    agent_1_in = []
    agent_1_out = []
    agent_2_in = []
    agent_2_out = []
    agent_ls = sorted(df.target_agent.unique())

    for ix, row in df.iterrows():
        if row["target_agent"] == agent_ls[0]:
            agent_0_in.append(get_confidence(str(row[f"{agent_ls[0]}_confidence"])))
            agent_1_out.append(get_confidence(str(row[f"{agent_ls[1]}_confidence"])))
            agent_2_out.append(get_confidence(str(row[f"{agent_ls[2]}_confidence"])))
        elif row["target_agent"] == agent_ls[1]:
            agent_0_out.append(get_confidence(str(row[f"{agent_ls[0]}_confidence"])))
            agent_1_in.append(get_confidence(str(row[f"{agent_ls[1]}_confidence"])))
            agent_2_out.append(get_confidence(str(row[f"{agent_ls[2]}_confidence"])))
        elif row["target_agent"] == agent_ls[2]:
            agent_0_out.append(get_confidence(str(row[f"{agent_ls[0]}_confidence"])))
            agent_1_out.append(get_confidence(str(row[f"{agent_ls[1]}_confidence"])))
            agent_2_in.append(get_confidence(str(row[f"{agent_ls[2]}_confidence"])))
    
    
    conf_in = {
        agent_ls[0]: agent_0_in,
        agent_ls[1]: agent_1_in,
        agent_ls[2]: agent_2_in
    }

    conf_out = {
        agent_ls[0]: agent_0_out,
        agent_ls[1]: agent_1_out,
        agent_ls[2]: agent_2_out
    }

    df_plot = {"agent": [], "confidence": [], "series": []}
    for agent in agent_ls:

        df_plot["confidence"].extend(conf_in[agent])
        df_plot["agent"].extend([agent]*len(conf_in[agent]))
        df_plot["series"].extend(["in domain"]*len(conf_in[agent]))

        df_plot["confidence"].extend(conf_out[agent])
        df_plot["agent"].extend([agent]*len(conf_out[agent]))
        df_plot["series"].extend(["out of domain"]*len(conf_out[agent]))

    df_plot = pd.DataFrame(df_plot)
    sns.boxplot(data = df_plot, x = "agent", y = "confidence", hue = "series")
    exp_name = list(args.experiment_name.split("_"))
    exp_name = exp_name[2:]
    exp_name_merge = ""
    for i in exp_name:
        exp_name_merge += "_"+i
    plt.title(f"{exp_name_merge}")
    plt.savefig(f"{log_dir}/boxplot_{exp_name_merge}.png")
    return


# function to train the model
def train(args, model, criterion, optimizer, train_dataloader, train_conf):
    model.train()

    total_loss, total_accuracy = 0, 0

    print("Training...")
    progress_bar = tqdm(range(len(train_dataloader)))

    # empty list to save model predictions
    total_preds=[]

    prev_index = 0
    cur_index = args.train_batch_size
    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()

        # get model predictions for the current batch
        train_confidence = train_conf[prev_index : cur_index]

        if type(model).__name__ == "BERT_BaseArch_Text":
            preds = model(sent_id, mask)
        elif type(model).__name__ == "BERT_BaseArch_Confidence":
            preds = model(train_confidence)
        else:
            preds = model(sent_id, mask, train_confidence)

        # update indeces for confidences
        temp_index = cur_index + args.train_batch_size
        if temp_index > len(train_conf):
            s = len(train_conf) - cur_index
            prev_index = cur_index
            cur_index = cur_index + s
        else:
            prev_index = cur_index
            cur_index = temp_index

        # compute the loss between actual and predicted values
        loss = criterion(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    return avg_loss


# compute the class weights
def compute_class_weight(context):
    n_i, n_j, n_k = 0, 0, 0
    for i in context.train_labels:
        if i == 0:
            n_i+=1
        elif i == 1:
            n_j+=1
        elif i == 2:
            n_k+=1
        
    j = [0,1,2]
    class_weights = []
    for i in j:
        n_classes = len(j)
        n_samples = len(context.train_labels)
        if i == 0:
            n_samplesj = n_i
        elif i == 1:
            n_samplesj = n_j
        else:
            n_samplesj = n_k
        class_weights.append(n_samples / (n_classes * n_samplesj))
    
    return class_weights 


# start training for all models
def train_model(args, context, model_class, split, log_dir):
    starttime = time.time()
    # to save all results
    all_eval = {}

    # to resume a model
    time.sleep(5)
    model = None
    # pass the pre-trained BERT to our defined architecture
    if model_class == "text":
        model = BERT_BaseArch_Text(context.domain_labels)
    elif model_class == "confidence":
        model = BERT_BaseArch_Confidence(context.domain_labels)
    elif model_class == "joined":
        model = BERT_BaseArch_Joint(context.domain_labels)
    
    print()
    print(f"Training started for {args.experiment_name}_{model_class}_{split}" )
    model = model.to(device)
    


    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(args.num_epochs): 
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.num_epochs))
        
        # load the BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length = args.max_seq_length)
        
        # define the optimizer
        optimizer = AdamW(model.parameters(), lr = args.learning_rate ) # best learning rate 5e-5
        criterion  = nn.CrossEntropyLoss()

        # set initial loss to infinite
        best_valid_loss = float('inf')
        
        # tokenize and encode sequences in the training set. Previously it was 25
        tokens_train = tokenizer.batch_encode_plus(
            context.train_text.tolist(),
            padding = "max_length", 
            max_length = args.max_seq_length,
            truncation = True
        )

        # convert lists to tensors
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(context.train_labels.tolist())

        # wrap tensors
        train_data = TensorDataset(train_seq, train_mask, train_y)

        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)

        # dataLoader for train set
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)

        # train model
        train_loss = train(args, model, criterion, optimizer, train_dataloader, context.train_conf)

        # save the best model
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), f"{log_dir}/{type(model).__name__}_{split}_{args.experiment_name}.pt")
        
        print(f'\nTraining Loss: {train_loss:.3f}')

        if  len(train_losses) > 0  and train_losses[-1] - train_loss < args.early_stopping_threshold:
            train_losses.append(train_loss)
            break 
        # append training and validation loss
        train_losses.append(train_loss)

    print("Training completed successfully!!\n")

    time.sleep(5)
    # start evaluation for module selection
    # load weights of the model
    path = f"{log_dir}/{type(model).__name__}_{split}_{args.experiment_name}.pt"

    model.load_state_dict(torch.load(path))
    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        context.test_text.tolist(),
        padding = "max_length", 
        max_length = args.max_seq_length,
        truncation=True
    )

    # convert lists to tensors
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    # get predictions for test data 
    with torch.no_grad():
        if type(model).__name__ == "BERT_BaseArch_Text":
            preds = model(test_seq.to(device), test_mask.to(device))
        elif type(model).__name__ == "BERT_BaseArch_Confidence":
            preds = model(context.test_conf.to(device))
        else:
            preds = model(test_seq.to(device), test_mask.to(device), context.test_conf.to(device))
    
    # detaching pred to cpu
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    
    target_names = list(context.domain_labels)

    final_report = classification_report(context.test_labels, preds, target_names = target_names, output_dict = True)

    all_eval[f"{args.experiment_name}_{model_class}_{split}"] = {
        "classification_report": final_report,
        "predictions": preds
    }

    print(type(model).__name__)
    print(classification_report(context.test_labels, preds, target_names = target_names))
    duration = (time.time()-starttime)
    return all_eval, duration


# detect domain and intent for modular settings
def modular_intent_detection(context, all_eval_result, args, split):
    ir = {}
    models = ["confidence", "text", "joined"]
    for key, value in all_eval_result.items():
        mod_ic = ModularIntentClassification()
        id = {}
        value_keys = list(value.keys())
        if "confidence" in value_keys[0]:
            model = models[0]
        elif "text" in value_keys[0]:
            model = models[1]
        elif "joined" in value_keys[0]:
            model = models[2]
            
        for i in range(len(value)):
            bot_type = context.domain_labels[i]
            f1_weighted, f1_macro, precision_weighted, precision_macro, recall_weighted, recall_macro = mod_ic.f1_intent_classification(context.test_df, y_pred = value[f"{args.experiment_name}_{model}_{split}"]["predictions"], bot_type = bot_type, data_context = context, args = args)

            id[bot_type] = {
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted
            }
        
        f1_weighted, f1_macro, precision_weighted, precision_macro, recall_weighted, recall_macro = mod_ic.f1_intent_classification(context.test_df, true_label = context.intent_labels, y_pred = value[f"{args.experiment_name}_{model}_{split}"]["predictions"], data_context = context, args = args)
        ir[key] = {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "mod_bots": id
        }
    return ir


# detect domain and intent for non-modular settings
def non_modular_intent_detection(context, args):
    non_mod = {}

    for i in range(3):
        bot_type = context.domain_labels[i]

        mod_ic = ModularIntentClassification()
        f1_weighted, f1_macro, precision_weighted, precision_macro, recall_weighted, recall_macro = mod_ic.f1_intent_classification(context.test_df, bot_type = bot_type, non_mod = True, data_context = context, args = args)

        non_mod[bot_type] = {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted
        }
    return non_mod


# extract results from confusion matrix
def extract_results(all_eval, args, split, ir):
    ms_scores = {}
    id_scores = {}

    for k , v in all_eval.items():
        key = None
        if k == "joined":
            key = "Joined Model"
        elif k == "confidence":
            key = "Confidence Model"
        elif k == "text":
            key = "Text model"
        ms_scores[key] = round(v[f"{args.experiment_name}_{k}_{split}"]["classification_report"]["weighted avg"]["f1-score"], 3)
        id_scores[key] =  round(ir[k]["f1_weighted"],3)

    ms_scores = dict(sorted(ms_scores.items(), key=lambda x: x[1], reverse=True))
    id_scores = dict(sorted(id_scores.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Module Selection Scores: {json.dumps(ms_scores, indent=4)}")
    print(f"Intent Detection scores: {json.dumps(id_scores, indent=4)}")
    
    return ms_scores, id_scores


# draw results for each split
def draw_results(args, log_dir, ir, all_eval_result, split):
    ms_scores, id_scores = extract_results(all_eval_result, args, split, ir) 
    
    plt.clf()
    models = list(ms_scores.keys())
    values = list(ms_scores.values())
    
    # Figure Size
    fig, ax = plt.subplots(figsize =(10, 8))
    
    # Horizontal Bar Plot
    ax.barh(models, values)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.8,
            alpha = 0.2)
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width(), i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 12, fontweight ='bold',
                color ='grey')

    exp_name = list(args.experiment_name.split("_"))
    exp_name = exp_name[2:]
    exp_name_merge = ""
    for i in exp_name:
        exp_name_merge += "_"+i
    plt.title(f"Module_selection_{exp_name_merge}(f1-scores)_split {split}")
    plt.savefig(f"{log_dir}/Module_selection_{exp_name_merge}(f1-scores) split {split}.png", dpi=70, bbox_inches='tight')
    
    
    plt.clf()
    # creating the dataset
    models = list(id_scores.keys())
    values = list(id_scores.values())
    
    # Figure Size
    fig, ax = plt.subplots(figsize =(10, 8))
    
    # Horizontal Bar Plot
    ax.barh(models, values)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.8,
            alpha = 0.2)
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width(), i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 12, fontweight ='bold',
                color ='grey')
    
    exp_name = list(args.experiment_name.split("_"))
    exp_name = exp_name[2:]
    exp_name_merge = ""
    for i in exp_name:
        exp_name_merge += "_"+i

    plt.title(f"Intent_detection_{exp_name_merge}(f1-scores) split {split}")
    plt.savefig(f"{log_dir}/Intent_detection_{exp_name_merge}(f1-scores) split {split}.png", dpi=70, bbox_inches='tight')

    return ms_scores, id_scores


# write all the results on result file
def write_on_log(ms_f1, id_f1, split, args, log_dir):
    for k, v in ms_f1.items():
        log_praefix = ""
        log_praefix += f", experiment={args.experiment_name}"
        log_praefix += f", model={k}"
        log_praefix += f", split={split}"
        log_praefix += ","
        write_results(log_dir, f"{log_praefix}f1_ms", v)

    for k, v in id_f1.items():
        log_praefix = ""
        log_praefix += f", experiment={args.experiment_name}"
        log_praefix += f", model={k}"
        log_praefix += f", split={split}"
        log_praefix += ","
        write_results(log_dir, f"{log_praefix}f1_id", v)
    return


# Write all preds into a csv file.
def write_preds_on_csv(args, split, context, all_eval_result, logs_dir):
    preds_csv = {
        "id": list(context.test_df.id),
        "hwu64_id": list(context.test_df.hwu64_id),
        "utterance": list(context.test_text),
        }
    predictions = []
    for k, v in all_eval_result.items():
        for k1, v1 in v.items():
            preds_csv[k1+"_preds"] = list(v1["predictions"]) 

    preds_csv_df = pd.DataFrame(preds_csv)
    preds_csv_df.to_csv(logs_dir+f"/{args.experiment_name}_{split}_preds.csv") 
    return


# run a single experiment that trains a module selector (if args.num_modules>1) and args.num_modules intent detectors
def single_experiment(args, log_dir, log_praefix = "", test_on_valid = False):
    if args.model == "all":
        models = ["confidence", "text", "joined"]  
    else:
        models = [args.model]

    if args.split == "all":
        splits = [0, 1, 2, 3, 4]
    else:
        splits = [int(args.split)]
    
    for split in splits:
        all_eval_result = {}
        data_context = load_modular_dataset(args, split)
        weights = torch.tensor(compute_class_weight(data_context), dtype = torch.float)
        # push to GPU
        weights = weights.to(device)
        for model in models:
            experiment_start_time = time.time()
            # train module selector
            logging.info(f"train module selector for {model} model")    
            eval_result, duration = train_model(args, data_context, model, split, log_dir)
            all_eval_result[model] = eval_result

        # compute acc_ms and f1_ms
        ir = modular_intent_detection(data_context, all_eval_result, args, split)
        ir["non_mod"] = non_modular_intent_detection(data_context, args)

        ms_f1, id_f1 = draw_results(args, log_dir, ir, all_eval_result, split)
        write_on_log(ms_f1, id_f1, split, args, log_dir)
        write_preds_on_csv(args, split, data_context, all_eval_result, log_dir)
    
    return 


# perform a hyperparameter search for optimal learning rate and batch size on hwu64 dataset for models bert and bert+adapter
def search_hyperparameter(args, log_dir):
    for batch_size in [8,16,32,64,128,256]:
        for learning_rate in [1e-5, 3e-5, 5e-5]:
            args.batch_size=batch_size
            args.learning_rate=learning_rate
            log_prefix=f"model=bert,learning_rate={learning_rate},batch_size={batch_size},"
            single_experiment(args, log_dir, log_praefix=log_prefix, test_on_valid=True)
    return


# perform all experiment for a single modular dataset
def full_experiment(args, log_dir):
    for dataset in ["inhomogenous", "homogenous_rasa", "homogenous_google_dialogflow", "homogenous_ibm_watson_assistant"]:
        if dataset == "inhomogenous":
            args.input_file = "model_experiments/datasets/modular_dialogsystem_inhomogenous_dataset.csv"
        else:
            args.input_file = f"model_experiments/datasets/modular_dialogsystem_dataset_{dataset}.csv"
        dataset = args.input_file.split("/")
        args.experiment_name = dataset[2].split(".")[0]
        create_box_plot(args, log_dir)
        logging.info(f"boxplot is created for {args.experiment_name}")
        
        single_experiment(args, log_dir)
    return


# sub-method of the main method
def main():
    try:
        starttime = time.time()
        args = read_args()
        dataset = args.input_file.split("/")
        if args.experiment == "single_experiment":
            args.experiment_name = dataset[2].split(".")[0]
            name = f"..num_modules={args.num_modules}"
            if args.num_train_samples > 0:
                name += "..subsample" 
            log_dir = init_experiments(args,"..single_experiment")
            create_box_plot(args, log_dir)
            logging.info(f"boxplot is created for {args.experiment_name}")
            single_experiment(args, log_dir)
        elif args.experiment == "full_experiment":
            log_dir = init_experiments(args, "..full_experiment")
            full_experiment(args, log_dir)
        elif args.experiment == "search_hyperparameter":
            args.experiment_name = dataset[2].split(".")[0]
            log_dir = init_experiments(args, "..search_hyperparameter")
            create_box_plot(args, log_dir)
            logging.info(f"boxplot is created for {args.experiment_name}")
            search_hyperparameter(args, log_dir)

        duration = (time.time()-starttime)/60
        logging.info(f"finished in {duration:.2f} minutes")

    except Exception as e:
        logging.exception(e)
    return


# the main method
if __name__ == '__main__':
    main()

