from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from dockstring.dataset import load_dataset
from tdc import Oracle

from acquisition_funcs.pareto import pareto_front

# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
known_smiles = ALL_SMILES[:100]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for various objectives
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")


def evaluate_objectives(smiles_list: list[str]) -> np.ndarray:
    """
    Given a list of N SMILES, return an Nx3 array A such that
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

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3])  # 3xN
    return out.T  # transpose, Nx3


known_Y = evaluate_objectives(known_smiles)
print(f"Known Y shape: {known_Y.shape}")
print(known_Y)

# Compute the Pareto front points from known_Y
pareto_mask = pareto_front(known_Y, maximize=True, deduplicate=True)
pareto_points = known_Y[pareto_mask]
non_pareto_points = known_Y[~pareto_mask]

print("Pareto front points:")
print(pareto_points)

# Plot the Pareto front and the other points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    non_pareto_points[:, 0],
    non_pareto_points[:, 1],
    non_pareto_points[:, 2],
    label="Non-Pareto Points",
    color="red",
    alpha=0.5,
)
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], label="Pareto Front Points", color="blue")

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_zlabel("Objective 3")
ax.set_title("Pareto Front Visualization in 3D")
ax.legend()

plt.show()
