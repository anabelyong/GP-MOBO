from __future__ import annotations

import logging
import random
import numpy as np
import torch
from typing import Callable
from dockstring.dataset import load_dataset
import gpytorch
import joblib
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from trf23.tanimoto_gp import TanimotoKernelGP, batch_predict_mu_std_numpy
from utils.utils import evaluate_single_objective

# Disable RDKit errors
rdBase.DisableLog("rdApp.error")

# Fingerprint parameters
FP_DIM = 2048
FP_RADIUS = 3
GP_MEAN = 0.00

# Setup logging
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

acq_opt_logger = logging.getLogger("acq_opt_logger")
bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.DEBUG)
bo_loop_logger.addHandler(stream_handler)

def smiles_to_fingerprint_arr(smiles_list: list[str]) -> np.array:
    """Convert SMILES to fingerprint array."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_DIM)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [mfpgen.GetCountFingerprintAsNumPy(m) for m in mols]
    return np.asarray(fps, dtype=float)

def get_gp_pred_on_smiles(
    smiles_list: list[str],
    model,
    device: torch.device,
    screen_batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Get GP predictions on SMILES."""
    fps = smiles_to_fingerprint_arr(smiles_list)
    return batch_predict_mu_std_numpy(model, fps, device=device, batch_size=screen_batch_size)

def run_tanimoto_gpbo(
    *,
    oracle: Callable[[list[str]], np.ndarray],  # Pass the oracle function
    smiles_bank: list[str],
    rng: random.Random,  # random number generator
    oracle_budget: int,
    num_start_samples: int = 10,
    max_bo_iter: int = 20,
    bo_batch_size: int = 1,
    max_heavy_atoms: int = 100,
    evaluate_single_objective: Callable[[list[str]], np.ndarray]  # Function to evaluate objectives
) -> None:
    bo_loop_logger.info("Starting BO loop...")

    # Canonicalize all smiles and remove duplicates
    bo_loop_logger.info("Canonicalizing all smiles")
    smiles_bank = list(set([Chem.CanonSmiles(s) for s in smiles_bank]))

    # Randomly choose initial smiles
    starting_population = rng.sample(smiles_bank, num_start_samples)
    starting_population_scores = oracle(starting_population)  # Ensure this returns an Nx1 array
    known_smiles_scores = {s: score.item() for s, score in zip(starting_population, starting_population_scores)}

    S_chosen = set()

    # Create query set by removing the starting population from smiles bank
    query_smiles = [s for s in smiles_bank if s not in known_smiles_scores]

    # Run BO loop
    for bo_iter in range(max_bo_iter):
        bo_loop_logger.info(f"Start BO iteration {bo_iter}. Dataset size={len(known_smiles_scores)}")

        # Featurize known smiles
        smiles_train = list(known_smiles_scores.keys())
        scores_train = np.array([known_smiles_scores[s] for s in smiles_train])  # Flatten to 1D array

        fp_train = smiles_to_fingerprint_arr(smiles_train)

        # Make GP and set hyperparameters
        torch.set_default_dtype(torch.float64)
        gp_model = TanimotoKernelGP(
            train_x=torch.as_tensor(fp_train),
            train_y=torch.as_tensor(scores_train).squeeze(),  # Ensure it is a 1D tensor
            kernel="T_MM",
            mean_obj=gpytorch.means.ConstantMean(),
        )
        gp_model.covar_module.raw_outputscale.requires_grad_(False)
        gp_model.mean_module.constant.requires_grad_(False)
        gp_model.likelihood.raw_noise.requires_grad_(False)
        gp_model.mean_module.constant.data.fill_(GP_MEAN)
        gp_model.covar_module.outputscale = 1.0
        gp_model.likelihood.noise = 1e-4
        gp_model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gp_model = gp_model.to(device)

        # Define acquisition function
        ucb_beta = 10.0 ** rng.uniform(-2, 0) if bo_iter < max_bo_iter else 0.0
        bo_loop_logger.info(f"UCB beta: {ucb_beta:.3f}")

        # Evaluate all query_smiles using the acquisition function
        acq_fn_values = {}
        for smiles in query_smiles:
            mu, std = get_gp_pred_on_smiles([smiles], gp_model, device)
            acq_value = mu + ucb_beta * std
            acq_fn_values[smiles] = acq_value.item()

        # Select the top SMILES based on the acquisition function value
        sorted_acq_fn_values = sorted(acq_fn_values.items(), key=lambda x: x[1], reverse=True)
        eval_batch = [s for s, _ in sorted_acq_fn_values[:bo_batch_size]]

        if not eval_batch:
            bo_loop_logger.info("No new SMILES selected.")
            continue

        # Log info about the batch
        mu_batch, std_batch = get_gp_pred_on_smiles(eval_batch, gp_model, device)
        eval_batch_acq_values = [acq_fn_values[s] for s in eval_batch]
        bo_loop_logger.debug(f"Eval batch SMILES: {eval_batch}")
        bo_loop_logger.debug(f"Eval batch acq values: {eval_batch_acq_values}")
        bo_loop_logger.debug(f"Eval batch mu: {mu_batch.tolist()}")
        bo_loop_logger.debug(f"Eval batch std: {std_batch.tolist()}")

        # Evaluate objectives for the chosen SMILES
        eval_batch_scores = evaluate_single_objective(eval_batch)
        eval_batch_scores = eval_batch_scores.flatten()  # Ensure this is a 1D array

        bo_loop_logger.debug(f"Eval batch scores: {eval_batch_scores}")
        known_smiles_scores.update({s: score for s, score in zip(eval_batch, eval_batch_scores)})

        # Print chosen SMILES and update dataset
        best_smiles = eval_batch[0]
        bo_loop_logger.info(f"Chosen SMILES: {best_smiles} with acquisition function value = {eval_batch_acq_values[0]}")
        bo_loop_logger.info(f"Value of chosen SMILES: {eval_batch_scores[0]}")
        bo_loop_logger.info(f"Updated dataset size: {len(known_smiles_scores)}")

        # Remove the evaluated smiles from the query set
        query_smiles = [s for s in query_smiles if s not in eval_batch]

        # Free up GPU memory for next iteration
        del gp_model
        torch.cuda.empty_cache()

    bo_loop_logger.info("Finished BO loop.")

if __name__ == "__main__":
    # Example SMILES bank (replace with your actual data)
    DOCKSTRING_DATASET = load_dataset()
    ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]

    # Running one experiment with random initial smiles
    initial_smiles = random.sample(ALL_SMILES, 10)
    known_smiles = initial_smiles.copy()
    query_smiles = [smiles for smiles in ALL_SMILES if smiles not in initial_smiles]

    # Evaluating the initial set of SMILES using the provided evaluate_single_objective function
    known_Y = evaluate_single_objective(known_smiles)
    print(f"Initial known_SMILES: {known_smiles}")
    print(f"Initial known_Y shape: {known_Y.shape}")
    print(f"Initial known_Y values: {known_Y}")

    # Run Bayesian Optimization Loop using evaluate_single_objective as the oracle
    run_tanimoto_gpbo(
        oracle=evaluate_single_objective,  # This is where the oracle is set
        smiles_bank=ALL_SMILES,
        rng=random.Random(),
        oracle_budget=50,
        num_start_samples=10,
        max_bo_iter=20,
        bo_batch_size=1,
        evaluate_single_objective=evaluate_single_objective,  # Already passed in run_tanimoto_gpbo
    )
