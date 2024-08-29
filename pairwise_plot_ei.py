import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from dockstring.dataset import load_dataset
from scipy import stats
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils import evaluate_single_objective

def expected_improvement(pred_means: np.array, pred_vars: np.array, y_best: float) -> np.array:
    """
    Calculate the expected improvement (EI) for each candidate point.

    Parameters:
    pred_means (np.array): Predicted means from the GP model.
    pred_vars (np.array): Predicted variances from the GP model.
    y_best (float): The best observed objective value.

    Returns:
    np.array: Expected improvement for each candidate.
    """
    std = np.sqrt(pred_vars)
    z = (pred_means - y_best) / std
    ei = (pred_means - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return np.maximum(ei, 1e-30)

def ei_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises):
    """
    Calculate the Expected Improvement (EI) for each query SMILES.
    
    Parameters:
    query_smiles (list[str]): List of SMILES strings to evaluate.
    known_smiles (list[str]): List of known SMILES strings.
    known_Y (np.array): Array of known objective values.
    gp_means (np.array): GP model mean parameters.
    gp_amplitudes (np.array): GP model amplitude parameters.
    gp_noises (np.array): GP model noise parameters.
    
    Returns:
    np.array: Array of EI values for each query SMILES.
    """
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )
    y_best = np.max(known_Y)
    ei_values = expected_improvement(pred_means, pred_vars, y_best)
    return ei_values, pred_means

def bayesian_optimization_loop(
    known_smiles,
    query_smiles,
    known_Y,
    gp_means,
    gp_amplitudes,
    gp_noises,
    n_iterations=20,
):
    S_chosen = set(known_smiles)
    results = []
    ei_values_list = []

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        max_acq = -np.inf
        best_smiles = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ei_values, _ = ei_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
            )
            ei_value = ei_values[0]
            if ei_value > max_acq:
                max_acq = ei_value
                best_smiles = smiles

        print(f"Max acquisition value (EI): {max_acq}")
        ei_values_list.append(max_acq)

        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_single_objective([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            chosen_value = new_Y[0][0]  # Extract the actual value of the chosen SMILES
            print(f"Chosen SMILES: {best_smiles} with EI value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
            # Append only the chosen value to the results
            results.append([iteration + 1, chosen_value])
        else:
            print("No new SMILES selected.")

    return known_smiles, known_Y, ei_values_list, results

def random_sampling_loop(
    known_smiles,
    query_smiles,
    known_Y,
    gp_means,
    gp_amplitudes,
    gp_noises,
    n_iterations=20,
):
    hypervolumes_rs = []
    acquisition_values_rs = []
    results_rs = []

    for iteration in range(n_iterations):
        print(f"Start Random Sampling iteration {iteration}. Dataset size={known_Y.shape}")

        best_smiles = np.random.choice(query_smiles)
        query_smiles.remove(best_smiles)
        new_Y = evaluate_single_objective([best_smiles])
        known_smiles.append(best_smiles)
        known_Y = np.vstack([known_Y, new_Y])
        print(f"Chosen SMILES: {best_smiles}")
        print(f"Value of chosen SMILES: {new_Y}")
        print(f"Updated dataset size: {known_Y.shape}")

        acquisition_values_rs.append(0)  # Random sampling does not use acquisition values
        results_rs.append([iteration, 0, best_smiles, new_Y[0][0]])

    return known_smiles, known_Y, acquisition_values_rs, results_rs

def plot_comparison(ei_values_bo, acquisition_values_rs):
    iterations = np.arange(1, len(ei_values_bo) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Expected Improvement (EI)", color="tab:blue")
    ax1.plot(iterations, ei_values_bo, "o-", color="tab:blue", label="EI BO")
    ax1.plot(iterations, acquisition_values_rs, "x-", color="tab:orange", label="Random Sampling")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    plt.title("Expected Improvement and Acquisition Function Value over Iterations")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.show()

def plot_pairwise_samples(known_Y_bo, known_Y_rs):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(known_Y_bo[:, 0], np.zeros_like(known_Y_bo[:, 0]), c="blue", label="BO Samples")
    axs[0].scatter(known_Y_rs[:, 0], np.zeros_like(known_Y_rs[:, 0]), c="green", label="RS Samples")
    axs[0].set_xlabel("Objective Value")
    axs[0].set_ylabel("")
    axs[0].legend()

    axs[1].scatter(known_Y_bo[:, 0], np.zeros_like(known_Y_bo[:, 0]), c="blue", label="BO Samples")
    axs[1].scatter(known_Y_rs[:, 0], np.zeros_like(known_Y_rs[:, 0]), c="green", label="RS Samples")
    axs[1].set_xlabel("Objective Value")
    axs[1].set_ylabel("")
    axs[1].legend()

    plt.suptitle("Pairwise Plots of Samples")
    plt.show()

def write_to_csv(results_bo, results_rs, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Iteration", "Chosen Smiles for Dockstring-MPO", "RS-MPO"])
        for bo, rs in zip(results_bo, results_rs):
            writer.writerow(bo + rs[1:])

if __name__ == "__main__":
    DOCKSTRING_DATASET = load_dataset()
    ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]

    # Running one experiment with random initial smiles
    initial_smiles = random.sample(ALL_SMILES, 20)
    known_smiles = initial_smiles.copy()
    query_smiles = [smiles for smiles in ALL_SMILES if smiles not in initial_smiles]

    known_Y = evaluate_single_objective(known_smiles)  # Single objective evaluation

    gp_means = np.asarray([0.0])
    gp_amplitudes = np.asarray([1.0])
    gp_noises = np.asarray([1e-4])

    # Bayesian Optimization Loop
    known_smiles_bo, known_Y_bo, ei_values_bo, results_bo = bayesian_optimization_loop(
        known_smiles=known_smiles.copy(),
        query_smiles=query_smiles.copy(),
        known_Y=known_Y.copy(),
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=20,
    )

    # Random Sampling Loop
    known_smiles_rs, known_Y_rs, acquisition_values_rs, results_rs = random_sampling_loop(
        known_smiles=known_smiles.copy(),
        query_smiles=query_smiles.copy(),
        known_Y=known_Y.copy(),
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=20,
    )

    # Plotting
    plot_pairwise_samples(known_Y_bo, known_Y_rs)
    plot_comparison(ei_values_bo, acquisition_values_rs)

    # Write results to CSV files
    write_to_csv(results_bo, results_rs, "dockstring_test.csv")

