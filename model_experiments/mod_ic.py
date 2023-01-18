import math 
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report 


class ModularIntentClassification:
    # parse and return the intent
    def get_intent(self, t_row):
        t_row = t_row[1:-1]
        t_row = ast.literal_eval(t_row)
        t_row = json.loads(t_row)

        return t_row["intent"]
    

    def get_modular_id(self, preds, true_labels, context, args):
        y_true=[]
        y_pred=[]
        inhomo_bot_ls = ["google_dialogflow", "rasa", "ibm_watson_assistant"]
        homo_bot_ls = ["agent_0", "agent_1", "agent_2"]
        inhomo = False
        if "inhomogenous" in list(args.experiment_name.split("_")):
            inhomo = True
        for ix, sample in enumerate(preds):
            bot = sample
            if inhomo == True and bot == 0:
                predicted_label = context.test_df[f"{inhomo_bot_ls[0]}_intent"].iloc[ix]
            elif inhomo == True and bot == 1:
                predicted_label = context.test_df[f"{inhomo_bot_ls[1]}_intent"].iloc[ix]
            elif inhomo == True and bot == 2:
                predicted_label = context.test_df[f"{inhomo_bot_ls[2]}_intent"].iloc[ix]
            elif inhomo==False and bot == 0:
                predicted_label = context.test_df[f"{homo_bot_ls[0]}_intent"].iloc[ix]
            elif inhomo == False and bot == 1:
                predicted_label = context.test_df[f"{homo_bot_ls[1]}_intent"].iloc[ix]
            elif inhomo == False and bot == 2:
                predicted_label = context.test_df[f"{homo_bot_ls[2]}_intent"].iloc[ix]
            true_label = None
            true_label = true_labels.iloc[ix]
            if predicted_label == None or predicted_label == "":
                predicted_label = "oos"
            
            if predicted_label == "oos":
                true_label = "oos"
            else:
                true_label = true_labels.iloc[ix]

            y_true.append(true_label)
            y_pred.append(predicted_label)

        return y_true, y_pred


    def get_modular_bots(self, df, preds, bot_type):
        y_true = []
        y_pred = []
        df_mod = df.copy()
        df_mod["selected_agent"] = torch.tensor(preds).tolist()
        df_mod = df_mod[df_mod["target_agent"] == bot_type]

        inhomo_bot_ls = ["google_dialogflow", "ibm_watson_assistant", "rasa"]
        homo_bot_ls = ["agent_0", "agent_1", "agent_2"]

        for ix, sample in df_mod.iterrows():
            if bot_type in inhomo_bot_ls and bot_type == "google_dialogflow" and sample["selected_agent"] == 0:
                predicted_label = sample[f"{bot_type}_intent"]
            elif bot_type in inhomo_bot_ls and bot_type == "rasa" and sample["selected_agent"] == 1:
                predicted_label = sample[f"{bot_type}_intent"]
            elif bot_type in inhomo_bot_ls and bot_type == "ibm_watson_assistant" and sample["selected_agent"] == 2:
                predicted_label = sample[f"{bot_type}_intent"]
            elif bot_type in homo_bot_ls and bot_type == "agent_0" and sample["selected_agent"] == 0:
                predicted_label = sample[f"{bot_type}_intent"]
            elif bot_type in homo_bot_ls and bot_type == "agent_1" and sample["selected_agent"] == 1:
                predicted_label = sample[f"{bot_type}_intent"]
            elif bot_type in homo_bot_ls and bot_type == "agent_2" and sample["selected_agent"] == 2:
                predicted_label = sample[f"{bot_type}_intent"]
            else:
                predicted_label = None

            true_label = None
            if predicted_label == None or predicted_label == "":
                predicted_label = "oos"
            
            if predicted_label == "oos":
                true_label = "oos"
            else:
                true_label = sample["true_intent"]
            y_true.append(true_label)
            y_pred.append(predicted_label)
        
        return y_true, y_pred
    

    def get_non_modular_bots(self, df, bot_type):
        y_true = []
        y_pred = []
        df_mod = df.copy()
        df_mod = df_mod[df_mod["target_agent"] == bot_type]

        inhomo_bot_ls = ["google_dialogflow", "ibm_watson_assistant", "rasa"]
        homo_bot_ls = ["agent_0", "agent_1", "agent_2"]
        for ix, sample in df_mod.iterrows():
            if bot_type:
                predicted_label = sample[f"{bot_type}_intent"]
            else:
                predicted_label = None

            true_label = None
            if predicted_label == None or predicted_label == "":
                predicted_label = "oos"
            
            if predicted_label == "oos":
                true_label = "oos"
            else:
                true_label = sample["true_intent"]
            y_true.append(true_label)
            y_pred.append(predicted_label)
        
        return y_true, y_pred

    # calculate intent prediction
    def get_predictions(self, df, true_labels, preds, bot_type, non_mod, context, args):
        if bot_type != None and non_mod == False:
            return self.get_modular_bots(df, preds, bot_type)
        elif bot_type != None and non_mod == True:
            return self.get_non_modular_bots(df, bot_type)
        return self.get_modular_id(preds, true_labels, context, args)

    # calculate f1-score of intent recognition
    def f1_intent_classification(self, df, true_label = None, y_pred = None, bot_type = None, non_mod = False, data_context = None, args = None ):
        y_true, y_pred = self.get_predictions(df, true_label, y_pred, bot_type, non_mod, data_context, args)

        precision_weighted = precision_score(y_true, y_pred, average = "weighted")
        precision_macro = precision_score(y_true, y_pred, average = "macro")
        recall_weighted = recall_score(y_true, y_pred, average = "weighted")
        recall_macro = recall_score(y_true, y_pred, average = "macro")
        f1_weighted = f1_score(y_true, y_pred, average = "weighted")
        f1_macro = f1_score(y_true, y_pred, average = "macro")

        return f1_weighted, f1_macro, precision_weighted, precision_macro, recall_weighted, recall_macro
