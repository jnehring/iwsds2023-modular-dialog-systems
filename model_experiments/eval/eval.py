import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report
import numpy as np
import os

agent_names = {
    "google_dialogflow": "GDF",
    "rasa": "RAS",
    "ibm_watson_assistant": "IWA"
}

models = ["confidence", "text", "joined"]

inhomogenous_agent_names = ["google_dialogflow", "rasa", "ibm_watson_assistant"]

model_names = {
    "confidence": "confidence",
    "text": "text",
    "joined": "text + confidence"
}

average = "weighted"

def confidence_boxplot():
    df = pd.read_csv("../../final_data/modular_dialogsystem_inhomogenous_dataset.csv")

    bxp = {"Confidence": [], "Module": [], "Series": []}
    for agent in df.target_agent.unique():
        conf = list(df[df.target_agent == agent][agent + "_confidence"])
        bxp["Confidence"].extend(list(conf))
        bxp["Module"].extend([agent] * len(conf))
        bxp["Series"].extend(["In domain"] * len(conf))

        conf = list(df[df.target_agent != agent][agent + "_confidence"])
        bxp["Confidence"].extend(list(conf))
        bxp["Module"].extend([agent] * len(conf))
        bxp["Series"].extend(["Out of domain"] * len(conf))

    bxp = pd.DataFrame(bxp)
    bxp.Module = bxp.Module.apply(lambda x: agent_names[x])
    plt.figure(figsize=(3, 3), dpi=80)
    sns.boxplot(data=bxp, x="Module", y="Confidence", hue="Series")
    plt.tight_layout()
    outfile = "outputs/confidence_boxplot.pdf"
    plt.savefig(outfile)
    print(f"created {outfile}")


def dataset_stats_table_inhomo():
    df = pd.read_csv("../../final_data/modular_dialogsystem_inhomogenous_dataset.csv")
    table = []
    for split in df.split.unique():
        for target_agent in df.target_agent.unique():
            _df = df[(df.split == split) & (df.target_agent == target_agent)]
            num_intents = len(_df.true_intent.unique())
            num_scenarios = len(_df.scenario.unique())
            num_samples = len(_df)
            # num_samples=f"{num_samples:,}"
            row = [split, agent_names[target_agent], num_intents, num_scenarios, num_samples]
            table.append(row)
    table = pd.DataFrame(table, columns=["Split", "Target agent", "Number of intents", "Number of scenarios",
                                         "Number of samples"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))

    outfile = "outputs/dataset_stats_table_inhomo.tex"
    table.to_latex(outfile, index=False)
    print(f"created {outfile}")

    sns.barplot(ax=ax1, data=table, x="Split", y="Number of samples", hue="Target agent")
    ax1.set_title("Number of samples")
    ax1.set_ylabel('')
    ax1.legend(loc='lower left')

    sns.barplot(ax=ax2, data=table, x="Split", y="Number of intents", hue="Target agent")
    ax2.set_title("Number of intents")
    ax2.set_ylabel('')

    sns.barplot(ax=ax3, data=table, x="Split", y="Number of scenarios", hue="Target agent")
    ax3.set_title("Number of scenarios")
    ax3.set_ylabel('')

    ax2.get_legend().remove()
    ax3.get_legend().remove()

    outfile = "outputs/dataset_stats_inhomo.pdf"
    plt.tight_layout()
    plt.savefig(outfile)
    print("created " + outfile)


def dataset_stats_table_homo():
    data = []

    for agent in agent_names:
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
        df = pd.read_csv(f"../../final_data/modular_dialogsystem_dataset_homogenous_{agent}.csv")

        for target_agent in sorted(list(df.target_agent.unique())):
            for split in df.split.unique():
                _df = df[(df.target_agent == target_agent) & (df.split == split)]
                num_intents = len(_df.true_intent.unique())
                num_scenarios = len(_df.scenario.unique())
                num_samples = len(_df)

                ta = "Agent " + target_agent[-1]
                row = [agent, ta, split, num_intents, num_scenarios, num_samples]
                data.append(row)
        table = pd.DataFrame(data,
                             columns=["Agent", "Target agent", "Split", "Number of intents", "Number of scenarios",
                                      "Number of samples"])
        sns.barplot(ax=ax1, data=table, x="Split", y="Number of samples", hue="Target agent")
        ax1.set_title("Number of samples")
        ax1.set_ylabel('')
        ax1.legend(loc='lower left')

        sns.barplot(ax=ax2, data=table, x="Split", y="Number of intents", hue="Target agent")
        ax2.set_title("Number of intents")
        ax2.set_ylabel('')

        sns.barplot(ax=ax3, data=table, x="Split", y="Number of scenarios", hue="Target agent")
        ax3.set_title("Number of scenarios")
        ax3.set_ylabel('')

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        outfile = f"outputs/dataset_stats_homo.pdf"
        plt.tight_layout()
        plt.savefig(outfile)
        print("created " + outfile)

        outfile = f"outputs/dataset_stats_homo.tex"
        table.to_latex(outfile, index=False)
        print("created " + outfile)
        break


def read_homogeneous():
    df_hom = []
    for agent in agent_names.keys():
        df = pd.read_csv(f"../../final_data/modular_dialogsystem_dataset_homogenous_{agent}.csv")
        df = df.sort_values(["hwu64_id"])
        df = df[df.dataset == "test"]

        df_joined = []
        for split in df.split.unique():
            dfs = df[df.split == split].copy().set_index("hwu64_id")
            dfp = pd.read_csv(f"predictions/modular_dialogsystem_dataset_homogenous_{agent}_{split}_preds.csv")
            dfp = dfp.sort_values(["hwu64_id"]).set_index("hwu64_id")

            assert len(dfs) == len(dfp)

            for c in dfp.columns[2:]:
                needle = f"modular_dialogsystem_dataset_homogenous_{agent}"

                if c[0:len(needle)] == needle:
                    new_name = c[len(needle) + 1:]
                    new_name = "ms_model_" + new_name[0:new_name.find("_")]
                    dfs[new_name] = dfp[c]

            df_joined.append(dfs)

        df_joined = pd.concat(df_joined)
        df_joined["dialog_system"] = agent
        df_hom.append(df_joined)

    df_hom = pd.concat(df_hom)
    # investigate_iwa(df_hom)
    # return

    compute_id_in_splits(df_hom)
    df_ms, df_id = get_scores_hom(df_hom)

    df_ms = df_ms.rename(columns={'F1_ms': "F1", 'Precision_ms': "Precision", 'Recall_ms': "Recall"})
    df_id = df_id.rename(columns={'F1_id': "F1", 'Precision_id': "Precision", 'Recall_id': "Recall"})

    df_ms["Task"] = "MS"
    df_id["Task"] = "ID"
    df = pd.concat((df_ms, df_id))
    df = df[['Task', 'Dialog System', 'Model', 'F1', 'Precision', 'Recall']]
    df["Model"] = df["Model"].apply(lambda x : model_names[x])

    outfile = "outputs/homogenous_scores.tex"
    df.to_latex(outfile, index=False)
    print("created " + outfile)


def compute_nonmod_f1():
    modules = {
        "google_dialogflow": "GDA",
        "ibm_watson_assistant": "IWA",
        "rasa": "RAS"
    }

    data = []
    for module in modules.keys():
        df = pd.read_csv(os.path.join("../../final_data/non-modular", module + ".csv"))
        y_true = list(df.true_intent)
        y_pred = list(df.predicted_intent)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=average)

        data.append([modules[module], f"{100 * f:.2f}%", f"{100 * p:.2f}%", f"{100 * r:.2f}%"])
    df = pd.DataFrame(data, columns=["module", "f1", "p", "r"])
    outfile = "outputs/nonmod.tex"
    df.to_latex(outfile, index=False)
    print("wrote " + outfile)


def get_scores_hom(df):
    # compute scores
    df.target_agent = df.target_agent.apply(lambda x: int(x[len("agent_"):]))

    all_intents = list(df.true_intent.unique())

    df_ms = []
    df_id = []
    #for dialog_system in ["rasa"]:
    for dialog_system in df.dialog_system.unique():
        #for model in ["text"]:
        for model in models:
            f1s_ms = []
            ps_ms = []
            rs_ms = []
            f1s_id = []
            ps_id = []
            rs_id = []
            for split in df.split.unique():
                dfs = df[(df.split == split) & (df.dialog_system == dialog_system)]

                y_true = dfs.target_agent
                y_pred = dfs["ms_model_" + model]
                f1s_ms.append(100 * f1_score(y_true, y_pred, average=average, zero_division=0))
                ps_ms.append(100 * precision_score(y_true, y_pred, average=average, zero_division=0))
                rs_ms.append(100 * recall_score(y_true, y_pred, average=average, zero_division=0))

                predicted_intents = []
                for ix, row in dfs.iterrows():
                    selected_module = row["ms_model_" + model]
                    intent = row[f"agent_{selected_module}_intent"]
                    predicted_intents.append(intent)

                f1s_id.append(100 * f1_score(dfs["true_intent"], predicted_intents, average=average, labels = all_intents, zero_division=0))
                ps_id.append(100 * precision_score(dfs["true_intent"], predicted_intents, average=average, labels = all_intents, zero_division=0))
                rs_id.append(100 * recall_score(dfs["true_intent"], predicted_intents, average=average, labels = all_intents, zero_division=0))

            row = [
                agent_names[dialog_system],
                model,
                f"{np.mean(f1s_ms):.2f}% ({np.std(f1s_ms):.2f})",
                f"{np.mean(ps_ms):.2f}% ({np.std(ps_ms):.2f})",
                f"{np.mean(rs_ms):.2f}% ({np.std(rs_ms):.2f})"
            ]

            df_ms.append(row)
            row = [
                agent_names[dialog_system],
                model,
                f"{np.mean(f1s_id):.2f}% ({np.std(f1s_id):.2f})",
                f"{np.mean(ps_id):.2f}% ({np.std(ps_id):.2f})",
                f"{np.mean(rs_id):.2f}% ({np.std(rs_id):.2f})"
            ]

            df_id.append(row)

    columns = ["Dialog System", "Model", "F1_ms", "Precision_ms", "Recall_ms"]
    df_ms = pd.DataFrame(df_ms, columns=columns)

    columns = ["Dialog System", "Model", "F1_id", "Precision_id", "Recall_id"]
    df_id = pd.DataFrame(df_id, columns=columns)

    return df_ms, df_id

def compute_id_in_splits(df_hom):

    stats = []

    for ds in df_hom.dialog_system.unique():
        f1_scores = []
        for target_agent in df_hom.target_agent.unique():
            for split in df_hom.split.unique():
                df = df_hom[
                    (df_hom.dialog_system == ds) &
                    (df_hom.target_agent == target_agent) &
                    (df_hom.split == split)
                ]
                f1 = f1_score(df.true_intent.astype(str), df[target_agent + "_intent"].astype(str), average=average)
                stats.append([ds, target_agent, split, f1])

    stats = pd.DataFrame(stats, columns = ["dialog_system", "target_agent", "split", "f1"]).sort_values(by=["f1"])
    outfile = "outputs/iwa_error_analysis.csv"
    stats.to_csv(outfile)
    print("wrote " + outfile)
# def investigate_iwa(df):
#
#     df.target_agent = df.target_agent.apply(lambda x: int(x[len("agent_"):]))
#
#     # df.dialog_system.unique()
#     all_intents = list(df.true_intent.unique())
#     data = []
#     for dialog_system in df.dialog_system.unique():
#         for model in ["text"]:
#             f1s_ms = []
#             ps_ms = []
#             rs_ms = []
#             f1s_id = []
#             ps_id = []
#             rs_id = []
#             for split in df.split.unique():
#                 dfsplit = df[(df.split == split) & (df.dialog_system == dialog_system)]
#
#                 for target_agent in dfsplit.target_agent.unique():
#
#                     dfta = dfsplit[dfsplit.target_agent == target_agent]
#                     predicted_intent_column = f"agent_{target_agent}_intent"
#                     true_intents = dfta.true_intent
#                     predicted_intents = dfta[predicted_intent_column].astype(str)
#
#                     print(len(true_intents), len(predicted_intents))
#
#                     # for pred in set(predicted_intents):
#                     #     if pred not in set(true_intents):
#                     #         print(dialog_system, pred)
#
#                     data.append([
#                         dialog_system,
#                         model,
#                         split,
#                         target_agent,
#                         100 * f1_score(true_intents, predicted_intents, labels=all_intents, average=average, zero_division=0),
#                         100 * precision_score(true_intents, predicted_intents, labels=all_intents, average=average, zero_division=0),
#                         100 * recall_score(true_intents, predicted_intents, labels=all_intents, average=average, zero_division=0)
#                     ])
#
#                     # if dialog_system == "ibm_watson_assistant" and target_agent == 2:
#                     #     print(classification_report(true_intents, predicted_intents))
#
#                     # y_true = dfs.target_agent
#                     #
#                     # print(dfs.target_agent.unique())
#                     # y_pred = dfs["ms_model_" + model]
#                     #
#                     # f1s_ms.append(100 * f1_score(y_true, y_pred, average=average, zero_division = 0))
#                     # ps_ms.append(100 * precision_score(y_true, y_pred, average=average, zero_division = 0))
#                     # rs_ms.append(100 * recall_score(y_true, y_pred, average=average, zero_division = 0))
#                     # #
#                     # #, average = average, zero_division = 0
#                     # predicted_intents = []
#                     # for ix, row in dfs.iterrows():
#                     #     selected_module = row["ms_model_" + model]
#                     #     intent = row[f"agent_{selected_module}_intent"]
#                     #     predicted_intents.append(intent)
#                     #
#                     # f1s_id.append(100 * f1_score(dfs["true_intent"], predicted_intents, labels = all_intents, average=average, zero_division = 0))
#                     # ps_id.append(100 * precision_score(dfs["true_intent"], predicted_intents, labels = all_intents, average=average, zero_division = 0))
#                     # rs_id.append(100 * recall_score(dfs["true_intent"], predicted_intents, labels = all_intents, average=average, zero_division = 0))
#                     # #
#
#     data = pd.DataFrame(data, columns=["dialog_system", "model", "split", "target_agent", "f1", "precision", "recall"])
#     data = data.sort_values(by=["dialog_system", "split", "target_agent"])
#     #print(data)

def get_scores_inhom(df):
    df_ms = []
    df_id = []

    for model in models:
        f1s_ms = []
        ps_ms = []
        rs_ms = []
        f1s_id = []
        ps_id = []
        rs_id = []
        for split in df.split.unique():
            dfs = df[(df.split == split)]

            y_true = list(dfs.target_agent)
            y_pred = list(dfs["ms_model_" + model])

            f1s_ms.append(100 * f1_score(y_true, y_pred, average=average))
            ps_ms.append(100 * precision_score(y_true, y_pred, average=average))
            rs_ms.append(100 * recall_score(y_true, y_pred, average=average))

            #            print(f"{100*f1_score(y_true, y_pred, average='micro'):.2f}")
            #            print(f"{100*precision_score(y_true, y_pred, average='micro'):.2f}")
            #            print(f"{100*recall_score(y_true, y_pred, average='micro'):.2f}")
            #            print()

            predicted_intents = []
            for ix, row in dfs.iterrows():
                selected_module = row["ms_model_" + model]
                predicted_intents.append(row[f"{selected_module}_intent"])
            f1s_id.append(100 * f1_score(dfs["true_intent"], predicted_intents, average=average))
            ps_id.append(100 * precision_score(dfs["true_intent"], predicted_intents, average=average))
            rs_id.append(100 * recall_score(dfs["true_intent"], predicted_intents, average=average))

        row = [
            model,
            f"{np.mean(f1s_ms):.2f}% ({np.std(f1s_ms):.2f})",
            f"{np.mean(ps_ms):.2f}% ({np.std(ps_ms):.2f})",
            f"{np.mean(rs_ms):.2f}% ({np.std(rs_ms):.2f})"]
        df_ms.append(row)
        row = [
            model,
            f"{np.mean(f1s_id):.2f}% ({np.std(f1s_id):.2f})",
            f"{np.mean(ps_id):.2f}% ({np.std(ps_id):.2f})",
            f"{np.mean(rs_id):.2f}% ({np.std(rs_id):.2f})"
        ]

        df_id.append(row)

    columns = ["Model", "F1_ms", "Precision_ms", "Recall_ms"]
    df_ms = pd.DataFrame(df_ms, columns=columns)

    columns = ["Model", "F1_ms", "Precision_id", "Recall_id"]
    df_id = pd.DataFrame(df_id, columns=columns)

    return df_ms, df_id


def read_inhomogenous():
    df_inhom = []
    df_inhom = pd.read_csv(f"../../final_data/modular_dialogsystem_inhomogenous_dataset.csv")
    df_inhom = df_inhom[df_inhom.dataset == "test"]
    df_joined = []
    for split in df_inhom.split.unique():
        dfs = df_inhom[df_inhom.split == split].copy().sort_values("hwu64_id").set_index("hwu64_id")

        dfp = pd.read_csv(f"predictions/modular_dialogsystem_inhomogenous_dataset_{split}_preds.csv")
        dfp = dfp.sort_values(["hwu64_id"]).set_index("hwu64_id")
        assert len(dfs) == len(dfp)

        for model in models:
            column = f"modular_dialogsystem_inhomogenous_dataset_{model}_{split}_preds"
            dfp[column] = dfp[column].apply(lambda x: inhomogenous_agent_names[x])
            dfs[f"ms_model_{model}"] = dfp[column]
        df_joined.append(dfs)

    df_joined = pd.concat(df_joined)

    df_ms, df_id = get_scores_inhom(df_joined)
    df_ms["Task"] = "MS"
    df_id["Task"] = "ID"

    df_ms = df_ms.rename(columns={'F1_ms': "F1", 'Precision_ms': "Precision", 'Recall_ms': "Recall", 'Precision_id': "Precision",
       'Recall_id': "Recall"})
    df_id = df_id.rename(columns={'F1_ms': "F1", 'Precision_ms': "Precision", 'Recall_ms': "Recall", 'Precision_id': "Precision",
       'Recall_id': "Recall"})

    df = pd.concat((df_ms, df_id))
    df = df[['Task', 'Model', 'F1', 'Precision', 'Recall']]

    outfile = "outputs/inhomogenous_scores.tex"
    df.to_latex(outfile, index=False)
    print("created " + outfile)

if __name__ == "__main__":
    confidence_boxplot()
    dataset_stats_table_homo()
    dataset_stats_table_inhomo()
    compute_nonmod_f1()

    read_homogeneous()
    read_inhomogenous()
