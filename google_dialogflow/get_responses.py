'''
Before running it make sure you have splitted the original data!!
to run it:
python -m google_dialogflow.get_responses
'''
import os
import pandas as pd
import time
from testbed.connectors.dialog_flow_interface import DialogFlowAPIInterface


# Create data for dialogflow and also create dialogflow chatbot
def create_dialogflow_bot(df, split_i):
    try:
        dialogFlow = DialogFlowAPIInterface.createInstance_self()
    except:
        print("Couldn't connect with Google Dialogflow!!")
        return None
    if dialogFlow:
        print(f"Data created for {split_i} dialogflow bot")
        dialogFlow.dynamic_setup(df, "mod_ir_self")
        
    return dialogFlow


# Create chatbot and produces outputs for each splits
def create_dialogflow_chatbots(df_hwu64, ith):
    if ith == 0:
        output_folder = f"google_dialogflow/tempfolder/dialogflow/"
    else:
        output_folder = f"google_dialogflow/tempfolder/dialogflow_{ith}/"
    splits=['0', '1' ,'2', '3', '4']
    
    splited_df = df_hwu64[df_hwu64.split.apply(lambda x : x in splits)]
    utterances = list(splited_df["utterance"])
    new_utterances  = []
    
    for i, utterance in enumerate(utterances):
        if len(utterance)>256:
            utterance = utterance[:256]
        new_utterances.append(utterance)

    splited_df["utterance"] = new_utterances

    new_intent = []
    for ix, row in splited_df.iterrows():
        t_i = row.intent.split(":")
        t_i_1 = t_i[1]
        t_i_1 = t_i_1.strip()
        
        t_i_0 = t_i[0]
        t_i_0 = t_i_0.strip()
        t_i = t_i_0 + "~" + t_i_1
        new_intent.append(t_i)
    splited_df["new_intent"] = new_intent

    for split in splits:
        print("="*50)
        print(split)
        i_splited_df = splited_df[splited_df.split == split]
        print(i_splited_df.count())
        if ith == 0:
            train_df = i_splited_df[(i_splited_df["dataset"] == "train_nlu") & (i_splited_df["target_agent"] == f"google_dialogflow") ]
        else:
            train_df = i_splited_df[(i_splited_df["dataset"] == "train_nlu") & (i_splited_df["target_agent"] == f"google_dialogflow_{ith}") ]

        new_folder=os.path.join(output_folder, str(split))
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        time.sleep(600)
        dialogflow = create_dialogflow_bot(train_df, split)
        
        if dialogflow == None:
            if ith == 0:
                print(f"Couldn't create google_dialogflow with split_{split} chatbot!!")
            else:
                print(f"Couldn't create google_dialogflow_{ith} with split_{split} chatbot!!")
        else:
            if ith == 0:
                print(f"Google_dialogflow with split_{split} chatbot created!!")
            else:
                print(f"Google_dialogflow_{ith} with split_{split} chatbot created!!")

        intent = []
        confidence = []
        count = 1
        count1 = 1
        time.sleep(60)
        for ix, row in i_splited_df.iterrows():
            print(row["utterance"])
            try:
                t, i, c = dialogflow.chat(row["utterance"])
            except:
                try:
                    time.sleep(60)
                    t, i, c = dialogflow.chat(row["utterance"])
                except:
                    raise
            print(i,c)
            if ix == 0:
                time.sleep(60)
            
            if "~" in i:
                i = i.split("~")
                i = i[0]+":"+i[1]
                
            intent.append(i)
            confidence.append(c)
            count += 1
            print(count)
            print(count1)
            print()
            if count == 100:
                time.sleep(65)
                count = 0
                count1+=1

        if ith == 0:
            outfile=f"google_dialogflow/tempfolder/dialogflow/" + split + f"/dataset.csv"
        else:
            outfile=f"google_dialogflow/tempfolder/dialogflow_{ith}/" + split + f"/dataset_dialogflow_{ith}.csv"

        i_splited_df["predicted_intent"] = intent
        i_splited_df["confidence"] = confidence
        i_splited_df.drop("new_intent", axis=1, inplace=True)
        i_splited_df.to_csv(outfile)   
    return


# Create dataframe from dataset
def get_gdf_df(dir):
    df = pd.read_csv(dir, encoding="utf-8")
    return df


# the main method
if __name__ == "__main__":
    for j in range(1,3):
        if j == 1:
            pd.options.mode.chained_assignment = None

            dir = "./datafolder/dataset_google_dialogflow.csv"
            df = get_gdf_df(dir)
            for i in range(1, 4):
                create_dialogflow_chatbots(df, i)
            pd.options.mode.chained_assignment = "warn"
        else:
            pd.options.mode.chained_assignment = None

            dir = "./datafolder/dataset.csv"
            df = get_gdf_df(dir)
            create_dialogflow_chatbots(df, 0)
            pd.options.mode.chained_assignment = "warn"

