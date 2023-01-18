import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report
import numpy as np

data = []
for dialog_system in ["google_dialogflow", "rasa", "ibm_watson_assistant"]:
    df = pd.read_csv(f"../datasets/modular_dialogsystem_dataset_homogenous_{dialog_system}.csv")

    for split in  df.split.unique():

        for target_agent in df.target_agent.unique():

            dfs = df[(df.split == split) & (df.dataset == "test") & (df.target_agent == target_agent)]

            train_intents = set(df[(df.split == split) & (df.dataset == "train_nlu") & (df.target_agent == target_agent)].true_intent)
            pred_intent = dfs[f"{target_agent}_intent"].astype(str)

            for pi in set(pred_intent):
                if pi not in train_intents:
                    data.append([dialog_system, split, target_agent, pi])

pd.set_option('display.max_rows', 1000)
data = pd.DataFrame(data, columns=["dialog_system", "split", "target_agent", "intent"])
print(data)
