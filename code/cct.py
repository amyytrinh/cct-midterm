import pandas as pd

url = "https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv"
df = pd.read_csv(url)

print(df.head())


