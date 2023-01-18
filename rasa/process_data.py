# for all splits, load the respective model, process all samples of the split and the results to a csv

'''
python -m rasa.process_data
'''

import json
import argparse
import pandas as pd
from tqdm import tqdm
from rasa.nlu.model import Trainer, Interpreter
import os


# generate all rasa models
def generate_rasa_models(rasa):
    for split in ["0", "1", "2", "3", "4"]:

        print("-"*20)
        print(split)
        print("-"*20)

        model_directory = os.path.join(f"/rasa/tempfolder/{rasa}/", split, "models/nlu/")
        rasa_interpreter = Interpreter.load(model_directory)
        if rasa == "rasa":
            df = pd.read_csv("/datafolder/dataset.csv")
        else:
            df = pd.read_csv("/datafolder/dataset_rasa.csv")
        df = df[df.split == split]

        pbar = tqdm(total = len(df))
        result_intent = []
        result_confidence = []
        for ix, sample in df.iterrows():
            result = rasa_interpreter.parse(sample["utterance"])
            result_intent.append(result["intent"]["name"])
            result_confidence.append(result["intent"]["confidence"])
            
            pbar.update(1)  
            
        outfile = f"/rasa/tempfolder/{rasa}/"+ split + f"/dataset_{rasa}.csv"
        df["predicted_intent"] = result_intent
        df["confidence"] = result_confidence
        df.to_csv(outfile)
    return


# the main method
if __name__ == "__main__":
    for rasa in ["rasa", "rasa_1", "rasa_2", "rasa_3"]:
        generate_rasa_models(rasa)
