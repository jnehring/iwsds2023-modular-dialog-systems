'''
to run it:
python -m model_experiments.test
'''


import pandas as pd
import numpy as np

res = pd.read_csv("./model_experiments/new_results.csv")

inhomo_f1ms_joined = []
inhomo_f1ms_text = []
inhomo_f1ms_conf = []
inhomo_f1id_joined = []
inhomo_f1id_text = []
inhomo_f1id_conf = []

homo_rasa_f1ms_joined = []
homo_rasa_f1ms_text = []
homo_rasa_f1ms_conf = []
homo_rasa_f1id_joined = []
homo_rasa_f1id_text = []
homo_rasa_f1id_conf = []

homo_gdf_f1ms_joined = []
homo_gdf_f1ms_text = []
homo_gdf_f1ms_conf = []
homo_gdf_f1id_joined = []
homo_gdf_f1id_text = []
homo_gdf_f1id_conf = []

homo_ibm_f1ms_joined = []
homo_ibm_f1ms_text = []
homo_ibm_f1ms_conf = []
homo_ibm_f1id_joined = []
homo_ibm_f1id_text = []
homo_ibm_f1id_conf = []

for k,v in res.iterrows():
    splits = v["key"].split(",")
    splits = splits[1:]
    experiment_name = splits[0].split("=")
    experiment_name = experiment_name[1]
    model_name = splits[1].split("=")
    model_name = model_name[1]
    if "inhomogenous" in experiment_name:
        if "Joined" in model_name:
            if "f1_ms" == splits[-1]:
                inhomo_f1ms_joined.append(v["value"])
            else:
                inhomo_f1id_joined.append(v["value"])
        elif "Text" in model_name:
            if "f1_ms" == splits[-1]:
                inhomo_f1ms_text.append(v["value"])
            else:
                inhomo_f1id_text.append(v["value"])
        elif "Confidence" in model_name:
            if "f1_ms" == splits[-1]:
                inhomo_f1ms_conf.append(v["value"])
            else:
                inhomo_f1id_conf.append(v["value"])
    elif "homogenous" in experiment_name and "rasa" in experiment_name:
        if "Joined" in model_name:
            if "f1_ms" == splits[-1]:
                homo_rasa_f1ms_joined.append(v["value"])
            else:
                homo_rasa_f1id_joined.append(v["value"])
        elif "Text" in model_name:
            if "f1_ms" == splits[-1]:
                homo_rasa_f1ms_text.append(v["value"])
            else:
                homo_rasa_f1id_text.append(v["value"])
        elif "Confidence" in model_name:
            if "f1_ms" == splits[-1]:
                homo_rasa_f1ms_conf.append(v["value"])
            else:
                homo_rasa_f1id_conf.append(v["value"])
    elif "homogenous" in experiment_name and "google_dialogflow" in experiment_name:
        if "Joined" in model_name:
            if "f1_ms" == splits[-1]:
                homo_gdf_f1ms_joined.append(v["value"])
            else:
                homo_gdf_f1id_joined.append(v["value"])
        elif "Text" in model_name:
            if "f1_ms" == splits[-1]:
                homo_gdf_f1ms_text.append(v["value"])
            else:
                homo_gdf_f1id_text.append(v["value"])
        elif "Confidence" in model_name:
            if "f1_ms" == splits[-1]:
                homo_gdf_f1ms_conf.append(v["value"])
            else:
                homo_gdf_f1id_conf.append(v["value"])
    elif "homogenous" in experiment_name and "ibm_watson_assistant" in experiment_name:
        if "Joined" in model_name:
            if "f1_ms" == splits[-1]:
                homo_ibm_f1ms_joined.append(v["value"])
            else:
                homo_ibm_f1id_joined.append(v["value"])
        elif "Text" in model_name:
            if "f1_ms" == splits[-1]:
                homo_ibm_f1ms_text.append(v["value"])
            else:
                homo_ibm_f1id_text.append(v["value"])
        elif "Confidence" in model_name:
            if "f1_ms" == splits[-1]:
                homo_ibm_f1ms_conf.append(v["value"])
            else:
                homo_ibm_f1id_conf.append(v["value"])  




inhomo_joined_ms_mean = round(np.mean(np.array(inhomo_f1ms_joined))*100, 2)
inhomo_text_ms_mean = round(np.mean(np.array(inhomo_f1ms_text))*100, 2)
inhomo_conf_ms_mean = round(np.mean(np.array(inhomo_f1ms_conf))*100, 2)
inhomo_joined_ms_var = round(np.std(np.array(inhomo_f1ms_joined))*100, 2)
inhomo_text_ms_var = round(np.std(np.array(inhomo_f1ms_text))*100, 2)
inhomo_conf_ms_var = round(np.std(np.array(inhomo_f1ms_conf))*100, 2)

inhomo_joined_id_mean = round(np.mean(np.array(inhomo_f1id_joined))*100, 2)
inhomo_text_id_mean = round(np.mean(np.array(inhomo_f1id_text))*100, 2)
inhomo_conf_id_mean = round(np.mean(np.array(inhomo_f1id_conf))*100, 2)
inhomo_joined_id_var = round(np.std(np.array(inhomo_f1id_joined))*100, 2)
inhomo_text_id_var = round(np.std(np.array(inhomo_f1id_text))*100, 2)
inhomo_conf_id_var = round(np.std(np.array(inhomo_f1id_conf))*100, 2)


homo_rasa_joined_ms_mean = round(np.mean(np.array(homo_rasa_f1ms_joined))*100, 2)
homo_rasa_text_ms_mean = round(np.mean(np.array(homo_rasa_f1ms_text))*100, 2)
homo_rasa_conf_ms_mean = round(np.mean(np.array(homo_rasa_f1ms_conf))*100, 2)
homo_rasa_joined_ms_var = round(np.std(np.array(homo_rasa_f1ms_joined))*100, 2)
homo_rasa_text_ms_var = round(np.std(np.array(homo_rasa_f1ms_text))*100, 2)
homo_rasa_conf_ms_var = round(np.std(np.array(homo_rasa_f1ms_conf))*100, 2)

homo_rasa_joined_id_mean = round(np.mean(np.array(homo_rasa_f1id_joined))*100, 2)
homo_rasa_text_id_mean = round(np.mean(np.array(homo_rasa_f1id_text))*100, 2)
homo_rasa_conf_id_mean = round(np.mean(np.array(homo_rasa_f1id_conf))*100, 2)
homo_rasa_joined_id_var = round(np.std(np.array(homo_rasa_f1id_joined))*100, 2)
homo_rasa_text_id_var = round(np.std(np.array(homo_rasa_f1id_text))*100, 2)
homo_rasa_conf_id_var = round(np.std(np.array(homo_rasa_f1id_conf))*100, 2)


homo_gdf_joined_ms_mean = round(np.mean(np.array(homo_gdf_f1ms_joined))*100, 2)
homo_gdf_text_ms_mean = round(np.mean(np.array(homo_gdf_f1ms_text))*100, 2)
homo_gdf_conf_ms_mean = round(np.mean(np.array(homo_gdf_f1ms_conf))*100, 2)
homo_gdf_joined_ms_var = round(np.std(np.array(homo_gdf_f1ms_joined))*100, 2)
homo_gdf_text_ms_var = round(np.std(np.array(homo_gdf_f1ms_text))*100, 2)
homo_gdf_conf_ms_var = round(np.std(np.array(homo_gdf_f1ms_conf))*100, 2)

homo_gdf_joined_id_mean = round(np.mean(np.array(homo_gdf_f1id_joined))*100, 2)
homo_gdf_text_id_mean = round(np.mean(np.array(homo_gdf_f1id_text))*100, 2)
homo_gdf_conf_id_mean = round(np.mean(np.array(homo_gdf_f1id_conf))*100, 2)
homo_gdf_joined_id_var = round(np.std(np.array(homo_gdf_f1id_joined))*100, 2)
homo_gdf_text_id_var = round(np.std(np.array(homo_gdf_f1id_text))*100, 2)
homo_gdf_conf_id_var = round(np.std(np.array(homo_gdf_f1id_conf))*100, 2)


homo_ibm_joined_ms_mean = round(np.mean(np.array(homo_ibm_f1ms_joined))*100, 2)
homo_ibm_text_ms_mean = round(np.mean(np.array(homo_ibm_f1ms_text))*100, 2)
homo_ibm_conf_ms_mean = round(np.mean(np.array(homo_ibm_f1ms_conf))*100, 2)
homo_ibm_joined_ms_var = round(np.std(np.array(homo_ibm_f1ms_joined))*100, 2)
homo_ibm_text_ms_var = round(np.std(np.array(homo_ibm_f1ms_text))*100, 2)
homo_ibm_conf_ms_var = round(np.std(np.array(homo_ibm_f1ms_conf))*100, 2)

homo_ibm_joined_id_mean = round(np.mean(np.array(homo_ibm_f1id_joined))*100, 2)
homo_ibm_text_id_mean = round(np.mean(np.array(homo_ibm_f1id_text))*100, 2)
homo_ibm_conf_id_mean = round(np.mean(np.array(homo_ibm_f1id_conf))*100, 2)
homo_ibm_joined_id_var = round(np.std(np.array(homo_ibm_f1id_joined))*100, 2)
homo_ibm_text_id_var = round(np.std(np.array(homo_ibm_f1id_text))*100, 2)
homo_ibm_conf_id_var = round(np.std(np.array(homo_ibm_f1id_conf))*100, 2)

df=[]

df.append(["homogeneous ID","", "", ""])
df.append([
    "rasa",
    f"{homo_rasa_conf_id_mean} ({homo_rasa_conf_id_var})", 
    f"{homo_rasa_text_id_mean} ({homo_rasa_text_id_var})", 
    f"{homo_rasa_joined_id_mean} ({homo_rasa_joined_id_var})", 
])
df.append([
    "gdf",
    f"{homo_gdf_conf_id_mean} ({homo_gdf_conf_id_var})", 
    f"{homo_gdf_text_id_mean} ({homo_gdf_text_id_var})", 
    f"{homo_gdf_joined_id_mean} ({homo_gdf_joined_id_var})", 
])
df.append([
    "ibm",
    f"{homo_ibm_conf_id_mean} ({homo_ibm_conf_id_var})", 
    f"{homo_ibm_text_id_mean} ({homo_ibm_text_id_var})", 
    f"{homo_ibm_joined_id_mean} ({homo_ibm_joined_id_var})", 
])
df.append(["homogeneous MS","", "", ""])
df.append([
    "rasa",
    f"{homo_rasa_conf_ms_mean} ({homo_rasa_conf_ms_var})", 
    f"{homo_rasa_text_ms_mean} ({homo_rasa_text_ms_var})", 
    f"{homo_rasa_joined_ms_mean} ({homo_rasa_joined_ms_var})", 
])
df.append([
    "gdf",
    f"{homo_gdf_conf_ms_mean} ({homo_gdf_conf_ms_var})", 
    f"{homo_gdf_text_ms_mean} ({homo_gdf_text_ms_var})", 
    f"{homo_gdf_joined_ms_mean} ({homo_gdf_joined_ms_var})", 
])
df.append([
    "ibm",
    f"{homo_ibm_conf_ms_mean} ({homo_ibm_conf_ms_var})", 
    f"{homo_ibm_text_ms_mean} ({homo_ibm_text_ms_var})", 
    f"{homo_ibm_joined_ms_mean} ({homo_ibm_joined_ms_var})", 
])
print("homogenous experiment resutls")
df=pd.DataFrame(df, columns=["agent", "conf", "text", "joined"])
print(df.to_latex(index=False))



df=[]

df.append([
    "Intent Detection",
    f"{inhomo_conf_id_mean} ({inhomo_conf_id_var})", 
    f"{inhomo_text_id_mean} ({inhomo_text_id_var})", 
    f"{inhomo_joined_id_mean} ({inhomo_joined_id_var})", 
])
df.append([
    "Module Selection",
    f"{inhomo_conf_ms_mean} ({inhomo_conf_ms_var})", 
    f"{inhomo_text_ms_mean} ({inhomo_text_ms_var})", 
    f"{inhomo_joined_ms_mean} ({inhomo_joined_ms_var})", 
])

df=pd.DataFrame(df, columns=["task", "conf", "text", "joined"])
print("inhomogenous experiment resutls")
print(df.to_latex(index=False))

# print(f"Inhomo MS mean of Joined: {inhomo_joined_ms_mean}, Text: {inhomo_text_ms_mean}, Confidence: {inhomo_conf_ms_mean}")
# print(f"Inhomo MS std of Joined: {inhomo_joined_ms_var}, Text: {inhomo_text_ms_var}, Confidence: {inhomo_conf_ms_var}")
# print()

# print(f"Inhomo ID mean of Joined: {inhomo_joined_id_mean}, Text: {inhomo_text_id_mean}, Confidence: {inhomo_conf_id_mean}")
# print(f"Inhomo ID std of Joined: {inhomo_joined_id_var}, Text: {inhomo_text_id_var}, Confidence: {inhomo_conf_id_var}")
# print()

# print(f"HomoRasa MS mean of Joined: {homo_rasa_joined_ms_mean}, Text: {homo_rasa_text_ms_mean}, Confidence: {homo_rasa_conf_ms_mean}")
# print(f"HomoRasa MS std of Joined: {homo_rasa_joined_ms_var}, Text: {homo_rasa_text_ms_var}, Confidence: {homo_rasa_conf_ms_var}")
# print()

# print(f"HomoRasa ID mean of Joined: {homo_rasa_joined_id_mean}, Text: {homo_rasa_text_id_mean}, Confidence: {homo_rasa_conf_id_mean}")
# print(f"HomoRasa ID std of Joined: {homo_rasa_joined_id_var}, Text: {homo_rasa_text_id_var}, Confidence: {homo_rasa_conf_id_var}")
# print()

# print(f"HomoGDF MS mean of Joined: {homo_gdf_joined_ms_mean}, Text: {homo_gdf_text_ms_mean}, Confidence: {homo_gdf_conf_ms_mean}")
# print(f"HomoGDF MS std of Joined: {homo_gdf_joined_ms_var}, Text: {homo_gdf_text_ms_var}, Confidence: {homo_gdf_conf_ms_var}")
# print()

# print(f"HomoGDF ID mean of Joined: {homo_gdf_joined_id_mean}, Text: {homo_gdf_text_id_mean}, Confidence: {homo_gdf_conf_id_mean}")
# print(f"HomoGDF ID std of Joined: {homo_gdf_joined_id_var}, Text: {homo_gdf_text_id_var}, Confidence: {homo_gdf_conf_id_var}")
# print()

# print(f"HomoIBMW MS mean of Joined: {homo_ibm_joined_ms_mean}, Text: {homo_ibm_text_ms_mean}, Confidence: {homo_ibm_conf_ms_mean}")
# print(f"HomoIBMW MS std of Joined: {homo_ibm_joined_ms_var}, Text: {homo_ibm_text_ms_var}, Confidence: {homo_ibm_conf_ms_var}")
# print()

# print(f"HomoIBMW ID mean of Joined: {homo_ibm_joined_id_mean}, Text: {homo_ibm_text_id_mean}, Confidence: {homo_ibm_conf_id_mean}")
# print(f"HomoIBMW ID std of Joined: {homo_ibm_joined_id_var}, Text: {homo_ibm_text_id_var}, Confidence: {homo_ibm_conf_id_var}")
# print()