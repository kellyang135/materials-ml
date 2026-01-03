computational mse project

predicts the bandgap of inorganic crystals from their chemical composition using open materials databases

## project overview

this repo explores bandgap prediction for inorganic crystals using:
+ composition-based featurization (e.g. element fractions)
+ supervised learning models (linear regression, tree-based models)
+ open datasets from the matminer ecosystem (e.g. MatBench)

there are two main parts:
+ a tiny local demo on a small synthetic dataset
+ a full experiment on a real benchmark dataset, intended to be run in a notebook environment 

## repo structure

- `01_load_data.py` / `02_features_and_model.py` – early exploration scripts 
- `03_tiny_dataset_model.py` – tiny demo using a synthetic bandgap dataset and composition features
- `03_real_dataset_model.py` – script targeting a real matminer benchmark dataset. however, bc of its size and library stack, this is best treated as reference code for a notebook-based experiment rather than something to run in every environment
- `check_setup.py`, `check_matminer.py` – environment sanity checks
- `data/` – local data
- `figures/` – generated plots (parity plots, feature importances, etc.)

## how to run locally

```bash
.venv/bin/python 03_tiny_dataset_model.py
```

this script:

- uses a tiny, hard-coded bandgap dataset of a few common semiconductors
- converts chemical formulas to `pymatgen` `Composition` objects
- generates composition features via `matminer.featurizers.composition.ElementFraction`
- trains a simple `LinearRegression` model
- prints mae / r² and saves a small parity plot to `figures/parity_tiny_bandgap_demo.png`


## how to run full matbench

suggested workflow:

1. open a new notebook (e.g. in colab)
2. install dependencies:

   ```python
   !pip install matminer pymatgen scikit-learn matplotlib
   ```

3. paste in the code outline from `notebooks/matbench_mp_gap_experiment.ipynb` and run the cells

## notebooks

- `notebooks/matbench_mp_gap_experiment.ipynb` – full experiment on matbench
  bandgap dataset (`matbench_mp_gap`). this notebook:
  - loads the dataset via `matminer.datasets.load_dataset`
  - constructs composition-based features using `ElementFraction`
  - trains and evaluates regression models
  - visualizes performance with a parity plot
