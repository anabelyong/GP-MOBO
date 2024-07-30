import numpy as np
from dockstring.dataset import load_dataset

from acquisition_funcs.ehvi import ehvi_acquisition
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives

# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
known_smiles = ALL_SMILES[:1000]
query_smiles = ALL_SMILES[100:1100]

# Evaluate objectives for known smiles
known_Y = evaluate_objectives(known_smiles)


# Define GP hyperparameters
gp_means = np.asarray([0.0, 0.0, 1.0, 0.0])
gp_amplitudes = np.asarray([1.0, 0.5, 1.0, 0.5])
gp_noises = np.asarray([1.0, 1e-4, 1e-1, 1e-2])

# Define a reference point for EHVI
reference_point = np.asarray([15.0, 1.0, 1.0, 1.0])

# BO loop
BO_known_smiles = list(known_smiles)
BO_known_Y = known_Y.copy()
for bo_iter in range(10):
    y_best = np.max(BO_known_Y, axis=0)  # best eval so far
    print(f"Start BO iter {bo_iter}. Dataset size={BO_known_Y.shape}. y_best={y_best}")

    # Make predictions
    mu_pred, var_pred = independent_tanimoto_gp_predict(
        query_smiles=ALL_SMILES,
        known_smiles=BO_known_smiles,
        known_Y=BO_known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )

    # Evaluate acquisition function (use analytic version)
    acq_fn_values = ehvi_acquisition(
        query_smiles=ALL_SMILES,
        known_smiles=BO_known_smiles,
        known_Y=BO_known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        reference_point=reference_point,
    )
    print(f"\tMax acquisition value: {max(acq_fn_values):.3g}")

    # Which SMILES maximizes the acquisition function value?
    chosen_smiles = None
    for chosen_i in np.argsort(-acq_fn_values):
        if ALL_SMILES[chosen_i] in BO_known_smiles:
            print(f"\tSMILES {chosen_i} with acq fn = {acq_fn_values[chosen_i]:.3g} is already known, skipping")
        else:
            chosen_smiles = ALL_SMILES[chosen_i]
            print(f"\tChose SMILES {chosen_i} with acq fn = {acq_fn_values[chosen_i]:.3g}")
            break

    # Evaluate SMILES
    new_y = evaluate_objectives([chosen_smiles])
    assert not np.any(np.isnan(new_y)), "NaN value detected in objective. Need to handle this case separately"
    print(f"\tValue of chosen SMILES: {new_y}")

    # Add to dataset
    BO_known_smiles.append(chosen_smiles)
    BO_known_Y = np.concatenate([BO_known_Y, new_y], axis=0)
