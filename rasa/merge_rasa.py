'''
Before running it make sure you have all the responses stored in the tempfolder under respective chatbot folder!!
to run it:
python -m rasa.merge_rasa
'''
import pandas as pd

# return a unique key for a single utterance
def make_key(text, intent, split):
    return str(text) + "_" + intent + "_" + str(split)


# merge all splits for three modules of IBN_Watson Assistant
def merge_rasa(df, ith):
    intents = []
    confidences = []
    
    for split in ["0", "1", "2", "3", "4"]:
        if ith == 0:
            splitfile = f"rasa/tempfolder/rasa_{ith}/" + split + f"/dataset_rasa.csv"
        else:
            splitfile = f"rasa/tempfolder/rasa_{ith}/" + split + f"/dataset_rasa_{ith}.csv"
        df_split = pd.read_csv(splitfile)

        indizes = df_split["Unnamed: 0.1"]
        additional_data = {}
        for i in range(len(indizes)):
            key = make_key(df_split.iloc[i]["Unnamed: 0.1"], df_split.iloc[i]["intent"], df_split.iloc[i]["split"])
            additional_data[key] = [df_split.iloc[i]["predicted_intent"], df_split.iloc[i]["confidence"]]
        
        t_df = df[df["split"] == split]
        for ix, row in t_df.iterrows():
            key = make_key(row["Unnamed: 0"], row["intent"], row["split"])
            intents.append(additional_data[key][0])
            confidences.append(additional_data[key][1])
        
        if ith == 0:
            print(f"rasa for split_{split} merged!!")
        else:
            print(f"rasa_{ith} for split_{split} merged!!")
    
    df["rasa_intent"] = intents
    df["rasa_confidence"] = confidences
    
    if ith == 0:
        df.to_csv(f"rasa/tempfolder/merged_dataset.csv")
    else:
        df.to_csv(f"rasa/tempfolder/merged_dataset_rasa_{ith}.csv")
    return


# the main method
if __name__ == "__main__":
    for j in range(1,3):
        if j == 1:
            df = pd.read_csv("datafolder/dataset_rasa.csv")
            df = df[df["split"] != "rasa"]
            for i in range(1,4):
                merge_rasa(df, i)
        else:
            df = pd.read_csv("datafolder/dataset.csv")
            df = df[df["split"] != "rasa"]
            merge_rasa(df, 0)