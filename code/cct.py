import pandas as pd
import numpy as np
import pymc as pm


def load_plant_knowledge():
    data_url = "../data/plant_knowledge.csv"
    df = pd.read_csv(data_url)
    df = df.drop(columns=["Informant"])
    
    #print(df)
    data = {
        "X": df.to_numpy(),
        "N": df.shape[0],
        "M": df.shape[1],
    }

    return data

