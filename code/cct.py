# Code provided in class repo was used for analyze_trace()
# ChatGPT was used to debug and modify code for analyze_trace()

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def load_plant_knowledge():
    data_url = "../data/plant_knowledge.csv"
    df = pd.read_csv(data_url)
    df = df.drop(columns=["Informant"])
    
    data = {
        "X": df.to_numpy(),  # Informant response
        "N": df.shape[0],    # Number of informants
        "M": df.shape[1],    # Number of questions
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


def analyze_trace(trace, data):
    X = data["X"]
    N = data["N"]
    M = data["M"]
    
    # Summary statistics for D (competence) and Z (consensus answers)
    summary = az.summary(trace, var_names=["D", "Z"])
    print("\nSummary Statistics:")
    print(summary)

    # Pair plot
    colors = ['#0000DD', '#DD0000', '#DD9500', '#00DD00']

    az.plot_pair(
        trace,
        var_names=["D", "Z"],
        coords={"D_dim_0": [0, 1, 2], "Z_dim_0": [0, 1, 2]},
        kind="kde",
        figsize=(10,10)
    )
    plt.suptitle("Posterior Pair Plot", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("posterior_pairplot.png")


    # Plot posterior distributions
    fig, ax = plt.subplots(figsize=(8, 8))
    az.plot_posterior(trace, var_names=["D"])
    plt.suptitle("Posterior for Informant Competence (D)")
    plt.tight_layout()
    plt.savefig("competence_posterior.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    az.plot_posterior(trace, var_names=["Z"])
    plt.suptitle("Posterior for Consensus Answers (Z)")
    plt.tight_layout()
    plt.savefig("consensus_posterior.png")
    plt.close()

    # Posterior mean of D and Z
    mean_D = trace.posterior["D"].mean(dim=["chain", "draw"]).values
    mean_Z = trace.posterior["Z"].mean(dim=["chain", "draw"]).values

    # Identify most and least competent informants
    most_competent = np.argmax(mean_D)
    least_competent = np.argmin(mean_D)

    print(f"\nMost competent informant: {most_competent} (mean D = {mean_D[most_competent]:.3f})")
    print(f"Least competent informant: {least_competent} (mean D = {mean_D[least_competent]:.3f})")

    # Estimate consensus answer key (mode of posterior mean, rounded to 0 or 1)
    consensus_key = (mean_Z > 0.5).astype(int)

    # Naive majority vote for each question
    majority_vote = (X.mean(axis=0) > 0.5).astype(int)

    # Compare the two answer keys
    matches = np.sum(consensus_key == majority_vote)
    print(f"\nConsensus vs Majority Agreement: {matches} / {M} questions match")

    return {
        "mean_D": mean_D,
        "mean_Z": mean_Z,
        "consensus_key": consensus_key,
        "majority_vote": majority_vote,
        "most_competent": most_competent,
        "least_competent": least_competent
    }


if __name__ == "__main__":
    data = load_plant_knowledge()
    print("Data loaded.")
    print(f"{data['N']} informants, {data['M']} questions.\n")

    trace = cct_model(data)
    print("Sampling complete.\n")

    results = analyze_trace(trace, data)
    print("\nAnalysis complete.")


