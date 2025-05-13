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

def cct_model(data):
    N = data["N"]
    M = data["M"]
    X = data["X"]
    
    with pm.Model() as model:
        # Priors
        D = pm.Uniform("D", lower=0.5, upper=1, shape=N) # priors for competence (D)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)  # priors for consensus answers (Z)
        
        D_reshaped = D[:, None]
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) 

        #Likelihood
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)  # Observed responses
        
        # Sampling
        trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True, target_accept=0.9)
    
    return trace





