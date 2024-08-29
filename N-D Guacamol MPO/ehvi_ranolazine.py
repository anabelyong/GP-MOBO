import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils_final import evaluate_ranol_objectives

def expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_front, N=1000):
    num_points, num_objectives = pred_means.shape
    ehvi_values = np.zeros(num_points)

    hv = Hypervolume(reference_point)
    current_hv = hv.compute(pareto_front)

    for i in range(num_points):
        mean = pred_means[i]
        var = pred_vars[i]
        cov = np.diag(var)

        # Monte Carlo integration
        samples = np.random.multivariate_normal(mean, cov, size=N)

        hvi = 0.0
        for sample in samples:
            augmented_pareto_front = np.vstack([pareto_front, sample])
            hv_sample = hv.compute(augmented_pareto_front)
            hvi += max(0, hv_sample - current_hv)

        ehvi_values[i] = hvi / N

    return ehvi_values

def ehvi_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, reference_point):
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )

    pareto_mask = pareto_front(known_Y)
    pareto_Y = known_Y[pareto_mask]

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y)

    return ehvi_values, pred_means

def bayesian_optimization_loop(
    known_smiles,
    query_smiles,
    known_Y,
    gp_means,
    gp_amplitudes,
    gp_noises,
    max_ref_point=None,
    scale=0.1,
    scale_max_ref_point=False,
    n_iterations=20,
):
    S_chosen = set()
    hypervolumes_bo = []
    acquisition_values = []
    results = []

    print(f"GP Means: {gp_means}")
    print(f"GP Amplitudes: {gp_amplitudes}")
    print(f"GP Noises: {gp_noises}\n")

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        max_acq = -np.inf
        best_smiles = None
        best_means = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ehvi_values, pred_means = ehvi_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
                reference_point=reference_point,
            )
            ehvi_value = ehvi_values[0]
            if ehvi_value > max_acq:
                max_acq = ehvi_value
                best_smiles = smiles
                best_means = pred_means[0]

        acquisition_values.append(max_acq)
        print(f"Max acquisition value: {max_acq}")
        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_ranol_objectives([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            print(f"Chosen SMILES: {best_smiles} with acquisition function value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
            # Store the value of the chosen SMILES (f1, f2, f3 from evaluation)
            f1, f2, f3, f4 = new_Y[0]  # Extract f1, f2, f3 directly from evaluation
            results.append([f1, f2, f3, f4])
        else:
            print("No new SMILES selected.")

        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_bo.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values, results

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
        hyperparam_Y = evaluate_ranol_objectives(hyperparam_smiles)
        training_Y = evaluate_ranol_objectives(training_smiles)

        # Calculate GP hyperparameters from the training set <- need to set epsilon value as noise was too small 
        epsilon = 1e-6
        gp_means = np.mean(hyperparam_Y, axis=0)
        gp_amplitudes = np.maximum(np.var(hyperparam_Y, axis=0), epsilon)
        gp_noises = np.maximum(np.var(hyperparam_Y, axis=0) * 0.1, epsilon)    # Assume 10% noise level of the variance

        # Run Bayesian Optimization Loop
        known_smiles_bo, known_Y_bo, hypervolumes_bo, acquisition_values_bo, experiment_results = bayesian_optimization_loop(
            known_smiles=training_smiles.copy(),
            query_smiles=query_smiles,
            known_Y=training_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=n_iterations,
        )

        # Append results with experiment number
        for iteration in range(n_iterations):
            f1, f2, f3, f4 = experiment_results[iteration]
            all_results.append([f"Experiment {experiment_num}", iteration + 1, f1, f2, f3, f4])

    return all_results

def write_results_to_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "BO Iteration", "EHVI-RANOL-MPO-F1", "EHVI-RANOL-MPO-F2", "EHVI-RANOL-MPO-F3", "EHVI-RANOL-MPO-F4"])
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    repeats = 3  # Number of experiments
    n_iterations = 20  # Number of BO iterations per experiment

    # Run the experiment and collect results
    results = run_experiment(repeats, n_iterations)

    # Write results to a CSV file
    write_results_to_csv(results, 'ehvi_bo_experiments_ranolazine_results.csv')
