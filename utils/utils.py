from pprint import pprint

import numpy as np
from dockstring.dataset import load_dataset
from tdc import Oracle

# 4 objectives instead <-- try smaller for now
# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
known_smiles = ALL_SMILES[:100]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for various objectives
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")
LOGP_ORACLE = Oracle("logp")


def evaluate_objectives(smiles_list: list[str]) -> np.ndarray:
    """
    Given a list of N SMILES, return an NxK array A such that
    A_{ij} is the jth objective function on the ith SMILES.

    NOTE: you might replace this implementation with an alternative
    implementation for your objective of interest.

    Our specific implementation uses the objectives above.
    Because it uses the dockstring dataset to look up PPARD values,
    it is only defined on SMILES in the dockstring dataset.

    Also, be careful of NaN values! Some docking scores might be NaN.
    These will need to be dealt with somehow.
    """
    # Initialize arrays for each objective
    f1 = np.array([-DOCKSTRING_DATASET["PPARD"].get(s, np.nan) for s in smiles_list])
    f2 = np.array(QED_ORACLE(smiles_list))
    f3 = np.array(CELECOXIB_ORACLE(smiles_list))
    # f4 = np.array(LOGP_ORACLE(smiles_list))

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")
    # print(f"f4 shape: {f4.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]
    # f4 = f4[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3])  # 4xN
    return out.T  # transpose, Nx4


known_Y = evaluate_objectives(known_smiles)
print(f"Known Y shape: {known_Y.shape}")
print(known_Y)
