import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# load the small dataset
df = pd.read_csv("data/bandgaps_small.csv")

# make a very small periodic table for the elements we use
#   Values: atomic_number, Pauling electronegativity (approx)
ELEMENTS = {
    "Si": {"Z": 14, "EN": 1.90},
    "Ge": {"Z": 32, "EN": 2.01},
    "Ga": {"Z": 31, "EN": 1.81},
    "As": {"Z": 33, "EN": 2.18},
    "N":  {"Z": 7,  "EN": 3.04},
    "O":  {"Z": 8,  "EN": 3.44},
    "Cd": {"Z": 48, "EN": 1.69},
    "Te": {"Z": 52, "EN": 2.10},
    "In": {"Z": 49, "EN": 1.78},
    "P":  {"Z": 15, "EN": 2.19},
    "Al": {"Z": 13, "EN": 1.61},
    "Zn": {"Z": 30, "EN": 1.65},
}

# manual composition lookup for our 10 formulas
#    for each formula, list (element, count) pairs
FORMULAS = {
    "Si":   [("Si", 1)],
    "Ge":   [("Ge", 1)],
    "GaAs": [("Ga", 1), ("As", 1)],
    "GaN":  [("Ga", 1), ("N", 1)],
    "SiO2": [("Si", 1), ("O", 2)],
    "CdTe": [("Cd", 1), ("Te", 1)],
    "InP":  [("In", 1), ("P", 1)],
    "AlN":  [("Al", 1), ("N", 1)],
    "ZnO":  [("Zn", 1), ("O", 1)],
    "GaP":  [("Ga", 1), ("P", 1)],
}

def featurize_formula(formula: str):
    """
    turn a formula string into a few numeric features
    for now, we look up its composition in FORMULAS
    """
    if formula not in FORMULAS:
        raise ValueError(f"Unknown formula {formula} in FORMULAS dict")

    components = FORMULAS[formula]

    total_atoms = sum(count for _, count in components)
    num_elements = len(components)

    # collect atomic numbers and electronegativities, weighted by count
    weighted_Z_sum = 0.0
    weighted_EN_sum = 0.0
    EN_values = []

    for element, count in components:
        props = ELEMENTS[element]
        weighted_Z_sum += props["Z"] * count
        weighted_EN_sum += props["EN"] * count
        EN_values.append(props["EN"])

    avg_Z = weighted_Z_sum / total_atoms
    avg_EN = weighted_EN_sum / total_atoms
    en_range = max(EN_values) - min(EN_values)

    return {
        "num_elements": num_elements,
        "avg_Z": avg_Z,
        "avg_EN": avg_EN,
        "en_range": en_range,
    }

# apply featurization to every row
feature_rows = []
for _, row in df.iterrows():
    f = featurize_formula(row["formula"])
    feature_rows.append(f)

features_df = pd.DataFrame(feature_rows)
print("Feature DataFrame:")
print(features_df, "\n")

# build X (features) and y (targets)
X = features_df
y = df["bandgap_ev"]

# train/test split (tiny dataset, so this is just for demonstration)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# train a simple Random Forest regressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Test set true bandgaps:", list(y_test.values))
print("Test set predicted bandgaps:", list(y_pred))
print(f"Mean absolute error on test set: {mae:.3f} eV")


# parity plot: predicted vs. true bandgap
plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred, color="blue", s=40, alpha=0.7, label="materials")

# diagonal line (perfect prediction)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="ideal")

plt.xlabel("True bandgap (eV)")
plt.ylabel("Predicted bandgap (eV)")
plt.title("Parity plot (small dataset)")
plt.legend()
plt.tight_layout()


import os

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/parity_small.png", dpi=150)
plt.show()
