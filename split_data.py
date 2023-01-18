'''
copy hwu64 in 5 splits and 4 one module only parts
run it:
python -m split_data
'''

import pandas as pd
import random


# fetch hwu64 dataset
def create_hwu64_dataframe():

    infile="datafolder/NLU-Data-Home-Domain-Annotated-All.csv"
    df_home=pd.read_csv(infile, sep=";")

    # drop samples that have no text
    df_home=df_home[df_home["answer"].notna()]

    df_home=pd.DataFrame({
        "utterance": df_home["answer"],
        "intent": df_home["intent"],
        "scenario": df_home["scenario"],
        "source": "hwu64",
        "additional_info": None
    })
    return df_home


# create all splits
def split_dataset(module_type, modules):
    random.seed(10)
    hwu64 = create_hwu64_dataframe()
    hwu64["intent"] = hwu64["scenario"] + ":" + hwu64["intent"]
    hwu64 = hwu64.drop(columns=["additional_info", "source"])

    n_splits = 5
   
    scenarios = list(hwu64.scenario.unique())

    df_result = []

    def subsample(df):
        n_samples = 100
        df = df.sort_values(["intent"])
        subsets = []
        for intent in df.intent.unique():
            subset = df[df.intent == intent]

            if len(subset) > n_samples:
                subset = subset.sample(n = n_samples)

            subsets.append(subset)

        subsets = pd.concat(subsets)
        return subsets

    for i_split in range(n_splits):
        scenario_mapping = {}
        for s in scenarios:

            module = modules[random.randint(0, len(modules)-1)]
            scenario_mapping[s] = module

        df_temp = hwu64.copy()

        df_temp = subsample(df_temp)
        df_temp["split"] = i_split
        df_temp["target_agent"] = df_temp.scenario.apply(lambda x : scenario_mapping[x])

        df_result.append(df_temp)

    for module in modules:
        df_temp = hwu64.copy()
        df_temp = subsample(df_temp)
        if module_type == "all":
            df_temp["split"] = module
        else:
            df_temp["split"] = module_type

        df_temp["target_agent"] = module
        df_result.append(df_temp)

    df_result = pd.concat(df_result)

    dataset = []
    datasets = ["train_nlu", "train_ms", "valid", "test"]
    for i in range(len(df_result)):
        di = random.randint(0, len(datasets)-1)
        dataset.append(datasets[di])

    df_result["dataset"] = dataset

    if module_type == "all":
        outfile = "datafolder/dataset.csv"
    else:
        outfile = F"datafolder/dataset_{module_type}.csv"
    df_result.to_csv(outfile)

    print(f"wrote {len(df_result)} rows to {outfile}")
    
    return


# the main method
if __name__ == "__main__":
    modules = {
        "all": ["rasa", "ïbm_watson_assistant", "google_dialogflow"],
        "google_dialogflow": ["google_dialogflow_1", "google_dialogflow_2", "google_dialogflow_3"],
        "rasa": ["rasa_1", "rasa_2", "rasa_3"],
        "ibm_watson_assistant": ["ibm_watson_assistant_1", "ibm_watson_assistant_2", "ibm_watson_assistant_3"]    
    }

    for k in modules.keys():
        split_dataset(k, modules[k])