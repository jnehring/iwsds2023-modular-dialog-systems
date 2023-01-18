import pandas as pd
import torch
import threading
import json
import transformers

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast
from ast import literal_eval

# parse and return the confidence
def get_confidence(t_row):
    ls = literal_eval(t_row)
    row = json.loads(str(ls))
    return row


# save all infos about training data in one place 
class ModularDataContext():

    def __init__(self):
        self.tokenizer = None
        self.split = None
        self.train_df = None
        self.test_df = None
        self.valid_df = None
        self.train_conf = None
        self.test_conf = None
        self.valid_conf = None
        self.train_text = None
        self.train_labels = None
        self.test_text = None
        self.test_labels = None
        self.test_ids = None
        self.valid_text = None
        self.valid_labels = None
        self.df = None
        self.domain_labels = None
        self.intent_labels = None
 


# dataset can be either train, test or valid
# use subsample to speed up processing by subsampling.
def load_modular_dataset(args, split, shuffle_train=True):
    context = ModularDataContext()
    context.split = split

    context.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length = args.max_seq_length)

    context.df = pd.read_csv(args.input_file)
    context.df = context.df.rename(columns={"Unnamed: 0": "id"})
    # context.domain_labels = list(context.df.target_agent.unique())
    inhomo_bot_ls = ["google_dialogflow", "rasa", "ibm_watson_assistant"]
    homo_bot_ls = ["agent_0", "agent_1", "agent_2"]

    if "inhomogenous" in list(args.experiment_name.split("_")):
        context.domain_labels = inhomo_bot_ls[:]
    else:
        context.domain_labels = homo_bot_ls[:]

    train_df = context.df[(context.df.dataset == "train_ms") & (context.df.split == split)]
    train_nlu_df = context.df[(context.df.dataset == "train_nlu") & (context.df.split == split)]
    context.train_df = pd.concat([train_df, train_nlu_df])

    # converting train set into tensors
    train_conf_all = []
    for i, row in context.train_df.iterrows():
        temp_ls = []
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[0]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[1]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[2]}_confidence"])))
        train_conf_all.append(temp_ls)

    context.train_text = context.train_df["utterance"]
    context.train_labels = context.train_df["target_agent"]
    context.train_conf = torch.tensor(train_conf_all, device='cuda', requires_grad = True)
    context.train_labels = context.train_labels.replace({context.domain_labels[0]:0, context.domain_labels[1]:1, context.domain_labels[2]:2})

    # converting test set into tensors
    context.test_df = context.df[(context.df.split == split) & (context.df.dataset == "test")]
    test_conf_all = []
    for i, row in context.test_df.iterrows():
        temp_ls = []
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[0]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[1]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[2]}_confidence"])))
        test_conf_all.append(temp_ls)
    
    context.test_text = context.test_df["utterance"]
    context.test_labels = context.test_df["target_agent"]
    context.test_conf = torch.tensor(test_conf_all, device='cuda', requires_grad = True)
    context.intent_labels = context.test_df.true_intent
    context.test_labels = context.test_labels.replace({context.domain_labels[0]:0, context.domain_labels[1]:1, context.domain_labels[2]:2})
    context.test_labels = torch.tensor(context.test_labels.tolist())
    
    # converting valid set into tensors
    context.valid_df = context.df[(context.df.split == split) & (context.df.dataset == "valid")]
    valid_conf_all = []
    for i, row in context.valid_df.iterrows():
        temp_ls = []
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[0]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[1]}_confidence"])))
        temp_ls.append(get_confidence(str(row[f"{context.domain_labels[2]}_confidence"])))
        valid_conf_all.append(temp_ls)
    
    context.valid_text = context.valid_df["utterance"]
    context.valid_labels = context.valid_df["target_agent"]
    context.valid_conf = torch.tensor(valid_conf_all, device='cuda', requires_grad = True)
    context.valid_labels = context.valid_labels.replace({context.domain_labels[0]:0, context.domain_labels[1]:1, context.domain_labels[2]:2})

    return context

