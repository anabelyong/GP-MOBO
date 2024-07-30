import numpy as np
from dockstring.dataset import load_dataset

from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives

# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
known_smiles = ALL_SMILES[:100]
query_smiles = ALL_SMILES[100:200]

# Evaluate objectives for known smiles
known_Y = evaluate_objectives(known_smiles)
print("Known Y shape:", known_Y.shape)
print(known_Y)

# TODO: Define GP hyperparameters <- check with Austin if amplitude and noise are manually made !
# optimize hyperparameters
# have a validation set maybe
gp_means = np.asarray([0.0, 0.0, 1.0, 0.0])
gp_amplitudes = np.asarray([1.0, 0.5, 1.0, 0.5])
gp_noises = np.asarray([1.0, 1e-4, 1e-1, 1e-2])

# Predict on a larger set of query smiles
pred_means, pred_vars = independent_tanimoto_gp_predict(
    query_smiles=query_smiles,
    known_smiles=known_smiles,
    known_Y=known_Y,
    gp_means=gp_means,
    gp_amplitudes=gp_amplitudes,
    gp_noises=gp_noises,
)

# Print results
print("Predicted Means:")
print(pred_means)
print("Predicted Variances:")
print(pred_vars)

# Optionally, print mean and variance for each objective
for i, smiles in enumerate(query_smiles):
    print(f"SMILES: {smiles}")
    print("Mean:", pred_means[i])
    print("Variance:", pred_vars[i])
    print("------")
