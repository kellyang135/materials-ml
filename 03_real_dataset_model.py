import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty, ElementFraction
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# load a bandgap dataset from matminer
# use a dataset that is actually available in this environment
# matbench_mp_gap has columns ['structure', 'gap pbe']
df = load_dataset("matbench_mp_gap")  # real benchmark dataset

print("Loaded dataset with shape:", df.shape)
print(df.head())

# clean up: drop rows with missing values in target or structure
target_col = "gap pbe"  # this is the band gap column name in this dataset
structure_col = "structure"
df = df[[structure_col, target_col]].dropna()
print("After dropping NaNs:", df.shape)

# for speed in this tutorial, subsample to an ultra-small size before featurization
# We only need enough data to demonstrate the workflow without overloading the machine
MAX_SAMPLES = 50
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=0)
    print(f"Subsampled to {len(df)} rows for featurization")

# convert Structure objects to Composition objects
df["composition"] = df[structure_col].apply(lambda s: s.composition)

# use a lighter composition featurizer to avoid heavy memory use
# ElementFraction gives the fractional amount of each element in the composition
ep = ElementFraction()

# this will add a moderate number of columns (one per element present in the data)
df_features = ep.featurize_dataframe(df, "composition", ignore_errors=True)

# build feature matrix X and target y
# all newly added columns from the featurizer will be after the original ones
feature_cols = [
    c
    for c in df_features.columns
    if c not in [structure_col, target_col, "composition"]
]
X = df_features[feature_cols]
y = df_features[target_col]

print("Number of features:", X.shape[1])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# train a very small, simple model to avoid stressing the system
# You can switch back to RandomForest later if this runs fine
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.3f} eV")
print(f"Test R^2: {r2:.3f}")

# (Optional) plotting is disabled to avoid any potential GPU/graphics issues
# Uncomment this block if everything runs stably and you want plots.
"""
# parity plot
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, s=5, alpha=0.5)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="ideal")
plt.xlabel("True bandgap (eV)")
plt.ylabel("Predicted bandgap (eV)")
plt.title("Bandgap prediction (matbench_mp_gap)")
plt.legend()
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/parity_matbench_mp_gap.png", dpi=150)
plt.show()

# feature importance (top 20) (not applicable to LinearRegression in the same way)
"""
