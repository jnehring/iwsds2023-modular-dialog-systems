'''
to run it:
python -m merge_all_datasets
'''

import pandas as pd
import json

# return a unique key for a single utterance
def make_key(text, intent, split):
    return str(text) + "_" + intent + "_" + str(split)


# create homogenous dataset for each agent
def create_homogenous_dataset(agent_name):
    root_data_path = f"datafolder/dataset_{agent_name}.csv" 
    sub_data_path = f"{agent_name}/tempfolder/merged_dataset_{agent_name}"

    df = pd.read_csv(root_data_path)
    df = df.rename(columns = {"Unnamed: 0" : "hwu64_id", "intent" : "true_intent"})
    df = df[df["split"] != agent_name]
    additional_data_ls = []
    for ith in range(1,4):
        additional_data = {}
        sub_df = pd.read_csv(sub_data_path+f"_{ith}.csv")
        indizes = sub_df["Unnamed: 0.1"]
        for i in range(len(indizes)):
            key = make_key(sub_df.iloc[i]["Unnamed: 0.1"], sub_df.iloc[i]["intent"], sub_df.iloc[i]["split"])
            additional_data[key] = [sub_df.iloc[i][f"{agent_name}_intent"], sub_df.iloc[i][f"{agent_name}_confidence"]]
        additional_data_ls.append(additional_data)

    print(f"additional data created for {agent_name}!!")
    for ith, additional_data in enumerate(additional_data_ls):
        intent_ls = []
        confidence_ls = []
        for ix, row in df.iterrows():
            key = make_key(row["hwu64_id"], row["true_intent"], row["split"])
            intent_ls.append(additional_data[key][0])
            confidence_ls.append(additional_data[key][1])

        df[f"agent_{ith}_intent"] = intent_ls
        df[f"agent_{ith}_confidence"] = confidence_ls

    df.to_csv(f"final_data/modular_dialogsystem_dataset_homogenous_{agent_name}.csv")
    print(f"homogenous modular dialog-system dataset created for {agent_name}!!")

    return


# chnage the target label name
def change_target_agent_names(module, path = f"final_data/modular_dialogsystem_dataset_homogenous_"):
    module_path = path+str(module)+".csv"
    df = pd.read_csv(module_path)
    tr_labels = []
    agent_labels = ["agent_0", "agent_1", "agent_2"]
    for ix, row in df.iterrows():
        if row.target_agent not in agent_labels:
            agent_no = int(row.target_agent[-1])-1
            tr_labels.append(agent_labels[agent_no])
    df.loc[:,"target_agent"] = tr_labels
    df.to_csv(module_path)

    return


# create inhomogenous datasets for all agent
def create_inhomogenous_dataset():
    agent_names = ["google_dialogflow", "ibm_watson_assistant", "rasa"]
    root_data_path = f"datafolder/dataset.csv" 
    df = pd.read_csv(root_data_path)
    df = df.rename(columns = {"Unnamed: 0" : "hwu64_id", "intent" : "true_intent"})

    for agent_name in agent_names:
        df = df[df["split"] != agent_name]
    
    for agent_name in agent_names:
        sub_data_path = f"{agent_name}/tempfolder/merged_dataset.csv"
        additional_data_ls = []
        for ith in range(1,4):
            additional_data = {}
            sub_df = pd.read_csv(sub_data_path)
            indizes = sub_df["Unnamed: 0.1"]
            for i in range(len(indizes)):
                key = make_key(sub_df.iloc[i]["Unnamed: 0.1"], sub_df.iloc[i]["intent"], sub_df.iloc[i]["split"])
                additional_data[key] = [sub_df.iloc[i][f"{agent_name}_intent"], sub_df.iloc[i][f"{agent_name}_confidence"]]
            additional_data_ls.append(additional_data)
        print(f"additional data created for {agent_name}!!")

        for ith, additional_data in enumerate(additional_data_ls):
            intent_ls = []
            confidence_ls = []
            for ix, row in df.iterrows():
                key = make_key(row["hwu64_id"], row["true_intent"], row["split"])
                intent_ls.append(additional_data[key][0])
                confidence_ls.append(additional_data[key][1])

            df[f"{agent_name}_intent"] = intent_ls
            df[f"{agent_name}_confidence"] = confidence_ls

    df.to_csv(f"final_data/modular_dialogsystem_inhomogenous_dataset.csv")
    print(f"inhomogenous modular dialog-system dataset created!!")

    return
    

# the main method
if __name__ == "__main__":
    modules = ["all", "google_dialogflow", "ibm_watson_assistant", "rasa"]
    for module in modules:
        if module == "all":
            create_inhomogenous_dataset()
        else:
            create_homogenous_dataset(module)
            change_target_agent_names(module)