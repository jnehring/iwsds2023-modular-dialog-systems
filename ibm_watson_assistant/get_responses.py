'''
Before running it make sure you have splitted the original data!!
to run it:
python -m ibm_watson_assistant.get_responses
'''
import pandas as pd
import os
import time
from testbed.connectors.watson_interface import WatsonAPIInterface


# create data for ibm_watson and also create ibm_watson chatbot
def create_ibm_watson_bot(df, split_i):
    try:
        ibm_watson = WatsonAPIInterface.createInstance()
    except:
        print("Couldn't connect with IBM Watson VA!!")
        return None
    if ibm_watson:
        print(f"Chatbot instance created for {split_i} ibm_watson bot")
        ibm_watson.setup_data(df) 
    return ibm_watson


# create directory for each split of ibm_watson assistant
def create_split_dir(split, ith):
    if ith == 0:
        output_folder = f"ibm_watson_assistant/tempfolder/ibm_watson"
    else:
        output_folder = f"ibm_watson_assistant/tempfolder/ibm_watson_{ith}"

    new_folder=os.path.join(output_folder, str(split))
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    return


# create output file with ibm_watson assistant responses for each split
def create_split_outfile(i_splited_df, split, intent, confidence, ith):
    if ith == 0:
        outfile=f"ibm_watson_assistant/tempfolder/ibm_watson/" + split + f"/dataset.csv" 
    else:
        outfile=f"ibm_watson_assistant/tempfolder/ibm_watson_{ith}/" + split + f"/dataset_ibm_watson_{ith}.csv"

    i_splited_df["predicted_intent"] = intent
    i_splited_df["confidence"] = confidence
    i_splited_df.drop("new_intent", axis=1, inplace=True)
    i_splited_df.to_csv(outfile)
    return


# main method to process all responses for ibm_watson assistant
def create_ibm_watson_responses(df_hwu64, ith):
    splits=['0', '1', '2', '3', '4']
    
    splited_df = df_hwu64[df_hwu64.split.apply(lambda x : x in splits)]

    new_intent = []
    for ix, row in splited_df.iterrows():
        t_i = row.intent.split(":")
        t_i_1 = t_i[1]
        t_i_1 = t_i_1.strip()
        
        t_i_0 = t_i[0]
        t_i_0 = t_i_0.strip()
        t_i = t_i_0 + "-" + t_i_1
        new_intent.append(t_i)
    splited_df["new_intent"] = new_intent

    if ith == 0:
        cache_path = f"ibm_watson_assistant/tempfolder/ibm_watson/cache/"
    else:
        cache_path = f"ibm_watson_assistant/tempfolder/ibm_watson_{ith}/cache/"

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    s = open(cache_path+"temp.txt", "a+", encoding="utf-8")
    data = ""
    try:
        data = pd.read_csv(cache_path+'temp.txt', sep=",")
    except ValueError as e:
        s.write("split,intent,confidence,counter")

    skip_split = -1
    skip_count = -1
    if len(data) and data.shape[0]>1:
        skip_split = list(data.split)[-1]
        skip_count = list(data.counter)[-1]-1

    for split in splits:
        if split<str(skip_split):
            continue

        print("="*50)
        print(split)
        i_splited_df = splited_df[splited_df.split == split]
        print(i_splited_df.count())
        if ith == 0:
            train_df = i_splited_df[(i_splited_df["dataset"] == "train_nlu") & (i_splited_df["target_agent"] == f"ibm_watson_assistant") ]
        else:
            train_df = i_splited_df[(i_splited_df["dataset"] == "train_nlu") & (i_splited_df["target_agent"] == f"ibm_watson_assistant_{ith}") ]
        ibm_watson = None
        ibm_watson = create_ibm_watson_bot(train_df, split)
        
        if ibm_watson == None:
            if ith == 0:
                print(f"Couldn't create ibm_watson_assistant with split_{split} chatbot!!")  
            else:
                print(f"Couldn't create ibm_watson_assistant_{ith} with split_{split} chatbot!!")
        else:
            if ith == 0:
                print(f"IBM_watson_assistant with split_{split} chatbot created!!")
            else:
                print(f"IBM_watson_assistant_{ith} with split_{split} chatbot created!!")

        intent = []
        confidence = []
        count = 0

        time.sleep(180)
        create_split_dir(split, ith)
        for ix, row in i_splited_df.iterrows():
            if count <= int(skip_count):
                intent.append(data.iloc[count].intent)
                confidence.append(data.iloc[count].confidence)
                
                if data.iloc[count].counter == 25609:
                    break
                count+=1
                continue

            skip_count = -1
            print(row["utterance"])
            i, c = ibm_watson.chat(row["utterance"])

            if "-" in i:
                i = i.split("-")
                i = i[0]+":"+i[1]
            
            print(i,c)
            print()
            intent.append(i)
            confidence.append(c)
            count += 1
            t = str(split)+","+str(intent[-1])+","+str(confidence[-1])+","+str(count)
            print(split, count)
            s.write("\n"+t)

        create_split_outfile(i_splited_df, split, intent, confidence, ith)
    return


# return the dataframe 
def get_df(dir):
    df = pd.read_csv(dir, encoding="utf-8")
    return df


# the main method
if __name__ == "__main__":
    for j in range(1,3):
        if j == 1:
            pd.options.mode.chained_assignment = None
            dir = "datafolder/dataset_ibm_watson_assistant.csv"
            df = get_df(dir)
            for i in range(1,4):
                create_ibm_watson_responses(df, i)
            pd.options.mode.chained_assignment = "warn"
        else:
            pd.options.mode.chained_assignment = None
            dir = "datafolder/dataset.csv"
            df = get_df(dir)
            create_ibm_watson_responses(df, 0)
            pd.options.mode.chained_assignment = "warn"           