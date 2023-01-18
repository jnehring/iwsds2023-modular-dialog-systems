'''
Before running it make sure you have all the responses stored in the tempfolder under respective chatbot folder!!
to run:
python -m google_dialogflow.merge_dialogflow
'''
import pandas as pd


# return a unique key for a single utterance
def make_key(text, intent, split):
    return str(text) + "_" + intent + "_" + str(split)


# merge all splits for three modules of Google Dialogflow Assistant
def merge_dialogflow(df, ith):
    intents = []
    confidences = []
    coun = 0
    for split in ["0", "1", "2", "3", "4"]:
        if ith == 0:
            splitfile = f"google_dialogflow/tempfolder/dialogflow/" + split + f"/dataset.csv"
        else:
            splitfile = f"google_dialogflow/tempfolder/dialogflow_{ith}/" + split + f"/dataset_dialogflow_{ith}.csv"
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
        print(f"dialogflow_{ith} for split_{split} merged!!")

    df["google_dialogflow_intent"] = intents
    df["google_dialogflow_confidence"] = confidences

    if ith == 0:
        df.to_csv(f"google_dialogflow/tempfolder/merged_dataset.csv")
    else:
        df.to_csv(f"google_dialogflow/tempfolder/merged_dataset_google_dialogflow_{ith}.csv")
    
    return


# the main method
if __name__ == "__main__":
    for j in range(1,3):
        if j == 1:
            df = pd.read_csv("datafolder/dataset_google_dialogflow.csv")
            df = df[df["split"] != "google_dialogflow" ]
            for i in range(1,4):
                merge_dialogflow(df, i)
        else:
            df = pd.read_csv("datafolder/dataset.csv")
            df = df[df["split"] != "google_dialogflow" ]
            merge_dialogflow(df, 0)