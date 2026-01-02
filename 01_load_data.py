import pandas as pd

# load the CSV
df = pd.read_csv("data/bandgaps_small.csv")

print("First few rows:")
print(df.head(), "\n")

print("DataFrame info:")
print(df.info(), "\n")

print("Basic statistics:")
print(df["bandgap_ev"].describe())


