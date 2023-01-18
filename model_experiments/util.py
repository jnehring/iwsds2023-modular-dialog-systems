
import argparse
import logging
from datetime import datetime
import os
import random
import torch
import numpy as np
import sys
import pandas as pd

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required = True, type = str, help = "Homogenous or inhomogenous file path.")
    parser.add_argument("--experiment", type = str, required = True, choices = ["single_experiment", "full_experiment", "search_hyperparameter"])
    parser.add_argument("--model", default = "all", type = str, help = "all|confidence|text|joined", choices = ["all", "confidence", "text", "joined"])
    parser.add_argument("--learning_rate", type = float, default = 3e-5, choices = [3e-4, 1e-4, 5e-5, 3e-5], help = "The initial learning rate for Adam.")
    parser.add_argument("--log_folder", type = str, default="model_experiments/logs")
    parser.add_argument("--train_batch_size", type = int, default = 32, choices = [8, 16, 32, 64, 128, 256])
    parser.add_argument("--test_batch_size", type = int, default = 32, choices = [8, 16, 32, 64, 128, 256])
    parser.add_argument("--output_dir", type = str, default = '')
    parser.add_argument("--split", type=str, default = "all", choices = ["all", "0", "1", "2", "3", "4"])
    parser.add_argument("--max_seq_length", type = int, default = 50)
    parser.add_argument("--num_modules", type = int, default = 3)
    parser.add_argument("--early_stopping_threshold", type = float, default = 0.002)
    parser.add_argument("--num_epochs", type = int, default = 1000)
    parser.add_argument("--logging_steps", type = int, default = 10)
    parser.add_argument("--num_train_samples", type = int, default = -1, help = "Subsample training data for debugging")
    parser.add_argument("--save_models", action = "store_true")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--experiment_name", type = str, default = "")
    
    return parser.parse_args()

# create the output folder
def make_log_folder(args, name):
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    my_date = datetime.now()

    folder_name = my_date.strftime('%Y-%m-%dT%H-%M-%S') + "_" + name

    if len(args.experiment_name) > 0:
        folder_name += "_" + args.experiment_name

    log_folder=os.path.join(args.log_folder, folder_name)
    os.mkdir(log_folder)
    return log_folder

# write a single store to the results logfile
results_df=None
def write_results(log_dir, key, value):

    logging.info(f"write result row key = {key}, value = {value}")
    results_file=os.path.join(log_dir, "results.csv")

    global results_df
    if results_df is None:
        results_df = pd.DataFrame(columns=["key", "value"])
    
    data = [key, value]
    results_df.loc[len(results_df)]=data

    results_df.to_csv(results_file)

# log to file and console
def create_logger(log_dir):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

# set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_experiments(args, experiment_name):
    log_dir = make_log_folder(args, experiment_name)
    logging.info(log_dir)
    create_logger(log_dir)
    set_seed(args.seed)

    command = " ".join(sys.argv)
    logging.info('''
 __  __           _       _                  _         _            
|  \/  |         | |     | |                | |       | |           
| \  / | ___   __| |_   _| | __ _ _ __    __| | __ _ _| |_ __ _ 
| |\/| |/ _ \ / _` | | | | |/ _` | '__|  / _` |/ _` | | __/ _` |
| |  | | (_) | (_| | |_| | | (_| | |    | (_| | (_| | | || (_| |
|_|  |_|\___/ \__,_|\__,_|_|\__,_|_|     \__,_|\__,_|  \__\__,_|
 ''')
    logging.info("start command: " + command)
    logging.info(f"experiment name {experiment_name}")
    logging.info(args)
    return log_dir

