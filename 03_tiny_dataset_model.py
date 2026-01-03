import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementFraction
from pymatgen.core import Composition
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Tiny synthetic bandgap dataset (illustrative, not precise)
# Each row is a semiconductor with an approximate bandgap in eV
DATA = {
    "formula": [
        "Si",    # silicon
        "Ge",    # germanium
        "GaAs",  # gallium arsenide
        "InP",   # indium phosphide
        "GaN",   # gallium nitride
        "AlN",   # aluminum nitride
        "ZnO",   # zinc oxide
        "CdTe",  # cadmium telluride
        "GaP",   # gallium phosphide
        "InAs",  # indium arsenide
    ],
    "bandgap": [
        1.1,
        0.7,
        1.4,
        1.3,
        3.4,
        6.0,
        3.3,
        1.5,
        2.3,
        0.36,
    ],
}


def main():
    # Put the data into a DataFrame
    df = pd.DataFrame(DATA)
    print("Tiny dataset:")
    print(df)

    # Convert formula strings to Composition objects
    df["composition"] = df["formula"].apply(Composition)

    # Use a light featurizer: fractional amount of each element in the composition
    featurizer = ElementFraction()
    df_feat = featurizer.featurize_dataframe(df, "composition", ignore_errors=True)

    # Build feature matrix X and target y
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in ["formula", "bandgap", "composition"]
    ]

    X = df_feat[feature_cols]
    y = df_feat["bandgap"]

    print("Number of features:", X.shape[1])

    # Train/test split (with such a tiny dataset this is just illustrative)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.3f} eV")
    print(f"Test R^2: {r2:.3f}")

    # Parity plot
    plt.figure(figsize=(4, 4))
    plt.scatter(y_test, y_pred, s=30, alpha=0.8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="ideal")
    plt.xlabel("True bandgap (eV)")
    plt.ylabel("Predicted bandgap (eV)")
    plt.title("Tiny bandgap demo (ElementFraction + LinearRegression)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/parity_tiny_bandgap_demo.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
