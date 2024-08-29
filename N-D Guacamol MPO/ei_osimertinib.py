import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils_final import evaluate_osim_MPO


def expected_improvement(pred_means: np.array, pred_vars: np.array, y_best: float) -> np.array:
    std = np.sqrt(pred_vars)
    z = (pred_means - y_best) / std
    ei = (pred_means - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return np.maximum(ei, 1e-30)


def ei_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises):
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
    S_chosen = set(known_smiles)  # Start with all known smiles
    ei_values_over_iterations = []
    best_values_over_iterations = []
    results = []

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        max_acq = -np.inf
        best_smiles = None
        best_mean = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ei_values, pred_means = ei_acquisition(
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
                best_mean = pred_means[0]

        ei_values_over_iterations.append(max_acq)
        print(f"Max acquisition value (EI): {max_acq}")

        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_osim_MPO([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            best_value = new_Y[0][0]  # Extract the value of the chosen SMILES
            best_values_over_iterations.append(best_value)
            print(f"Chosen SMILES: {best_smiles} with EI value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
            # Store the value of the chosen SMILES (actual evaluation)
            results.append([iteration + 1, best_value])
        else:
            print("No new SMILES selected.")
            best_values_over_iterations.append(np.max(known_Y))

    return known_smiles, known_Y, ei_values_over_iterations, best_values_over_iterations, results


def plot_results(ei_values, best_values):
    iterations = np.arange(1, len(ei_values) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Expected Improvement", color="tab:blue")
    ax1.plot(iterations, ei_values, "o-", color="tab:blue", label="Expected Improvement")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Best SMILES Value", color="tab:red")
    ax2.plot(iterations, best_values, "o-", color="tab:red", label="Best SMILES Value")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Expected Improvement and Best SMILES Value over BO Iterations")
    plt.show()


def write_to_csv(results, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Iteration", "Chosen Smiles for Osimertinib-MPO"])
        for row in results:
            writer.writerow(row)


def run_experiment(repeats, n_iterations):
    all_results = []

    for experiment_num in range(1, repeats + 1):
        print(f"\nStarting Experiment {experiment_num}...\n")

        guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
        guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
        ALL_SMILES = guacamol_dataset["smiles"].tolist()[:11_000]

        hyperparam_smiles = ALL_SMILES[10_000:]
        ALL_SMILES = ALL_SMILES[:10_000]
        random.shuffle(ALL_SMILES)
        training_smiles = ALL_SMILES[:20]
        query_smiles = ALL_SMILES[20:10_000]

        # Calculate objectives for training smiles
        hyperparam_Y = evaluate_osim_MPO(hyperparam_smiles)
        training_Y = evaluate_osim_MPO(training_smiles)

        # Calculate GP hyperparameters from the training set
        gp_means = np.mean(hyperparam_Y, axis=0)
        gp_amplitudes = np.var(hyperparam_Y, axis=0)
        gp_noises = np.var(hyperparam_Y, axis=0) * 0.1  # Assume 10% noise level of the variance

        # Print the GP hyperparameters before starting the BO loop
        print("Calculated GP Hyperparameters:")
        print(f"GP Means: {gp_means}")
        print(f"GP Amplitudes: {gp_amplitudes}")
        print(f"GP Noises: {gp_noises}\n")

        # Start with all 1000 training smiles in the known_smiles set
        known_smiles = training_smiles.copy()
        known_Y = training_Y.copy()

        # Bayesian Optimization Loop
        known_smiles_bo, known_Y_bo, ei_values_bo, best_values_bo, experiment_results = bayesian_optimization_loop(
            known_smiles=known_smiles,
            query_smiles=query_smiles,
            known_Y=known_Y,
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=n_iterations,
        )

        # Append results with experiment number
        for iteration in range(n_iterations):
            all_results.append([f"Experiment {experiment_num}"] + experiment_results[iteration])

    return all_results


if __name__ == "__main__":
    repeats = 3  # Number of experiments
    n_iterations = 20  # Number of BO iterations per experiment

    # Run the experiment and collect results
    results = run_experiment(repeats, n_iterations)

    # Write results to a CSV file
    write_to_csv(results, "ei_bo_experiments_osimertinib_results.csv")
