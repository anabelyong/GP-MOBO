import random
from dockstring.dataset import load_dataset
from scipy.stats import gmean
import numpy as np
from tdc import Oracle

# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]

# Define oracles
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")

# Define the function to evaluate the geometric mean of objectives
def get_geometric_mean(smiles: str) -> float:
    """
    Calculate the geometric mean of the three oracles for a single SMILES string.

    Parameters:
    smiles (str): A SMILES string to evaluate.

    Returns:
    float: The geometric mean of the oracle evaluations.
    """
    f1 = -DOCKSTRING_DATASET["PPARD"].get(smiles, np.nan)  # Use negative since PPARD values are usually scores
    f2 = QED_ORACLE(smiles)
    f3 = CELECOXIB_ORACLE(smiles)

    values = [f1, f2, f3]

    # Filter out any NaN values to avoid issues in geometric mean calculation
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        raise ValueError("All oracle values are NaN.")

    return gmean(values)

# Evaluate the objectives for the first 10,000 SMILES
geometric_means = [(smiles, get_geometric_mean(smiles)) for smiles in ALL_SMILES]

# Sort by geometric mean and get the top 20 SMILES
top_20_smiles = sorted(geometric_means, key=lambda x: x[1], reverse=True)[:20]
print(top_20_smiles)
