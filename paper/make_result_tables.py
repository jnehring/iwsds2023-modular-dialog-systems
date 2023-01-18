import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate')
    choices=["inhom"]
    parser.add_argument('-experiment', type=str, choices=choices, help=f'Which analysis? Choice of {choices}')
    args = parser.parse_args()

    datafiles = {
        "gdf": "final_data/modular_dialogsystem_dataset_homogenous_google_dialogflow.csv",
        "iwa": "final_data/modular_dialogsystem_dataset_homogenous_ibm_watson_assistant.csv",
        "ras": "final_data/modular_dialogsystem_dataset_homogenous_rasa.csv",
        "inhom": "final_data/modular_dialogsystem_inhomogenous_dataset.csv"
    }

    if args.experiment == "inhom":
        df=pd.read_csv(datafiles["inhom"])
        print(df.columns)