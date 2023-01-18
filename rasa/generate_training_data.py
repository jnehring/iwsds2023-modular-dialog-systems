# generate training data for rasas intent detection

'''
python -m rasa.generate_training_data
'''
import json
import os
from shutil import copyfile
import pandas as pd


# generate training data for all rasa module
def generate_rasa_training_data(rasa):
    output_folder = f"rasa/tempfolder/{rasa}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if rasa == "rasa":
        df = pd.read_csv("datafolder/dataset.csv")
    else:
        df = pd.read_csv("datafolder/dataset_rasa.csv")

    for split in df.split.unique():
        df_split=df[(df.split == split) & (df.target_agent == rasa) & (df.dataset == "train_nlu")]
        if len(df_split)==0:
            continue

        examples=[]

        for ix, sample in df_split[["utterance", "intent"]].iterrows():

            sample={
                "entities": [],
                "text": sample["utterance"],
                "intent": sample["intent"]
            }
            examples.append(sample)


        rasa_data = {"rasa_nlu_data": {"common_examples": examples}}

        new_folder=os.path.join(output_folder, split, "data")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        outfile=os.path.join(new_folder, "train.json")
        f = open(outfile, "w")
        f.write(json.dumps(rasa_data, indent=4))
        f.close()
    return

# the main method
if __name__ == "__main__":
    for rasa in ["rasa", "rasa_1", "rasa_2", "rasa_3"]:
        generate_rasa_training_data(str(rasa))