import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils_final import evaluate_perin_objectives

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

def ehvi_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, reference_point, N=1000):
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

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y, N=N)

    return ehvi_values, pred_means

def ehvi_error_analysis(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, reference_point, N_values, num_runs=50):
    std_devs = []

    for N in N_values:
        ehvi_samples = []

        for _ in range(num_runs):
            ehvi_values, _ = ehvi_acquisition(
                query_smiles=query_smiles,
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
                reference_point=reference_point,
                N=N
            )
            avg_ehvi = np.mean(ehvi_values)
            ehvi_samples.append(avg_ehvi)

        std_dev = np.std(ehvi_samples)
        std_devs.append(std_dev)

    return std_devs

def run_error_analysis_experiment():
    N_values = [1, 10, 100, 500, 1000, 10000]
    num_runs = 50  # Number of independent runs for each N

    guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
    guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
    ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]

    random.shuffle(ALL_SMILES)
    training_smiles = ALL_SMILES[:10]
    query_smiles = ALL_SMILES[10:10_000]

    # Calculate objectives for training smiles
    training_Y = evaluate_perin_objectives(training_smiles)

    # Calculate GP hyperparameters from the training set
    gp_means = np.asarray([0.0, 0.0])
    gp_amplitudes = np.asarray([1.0, 1.0])
    gp_noises = np.asarray([1e-4, 1e-4])

    # Set the reference point
    reference_point = infer_reference_point(
        training_Y, scale=0.1, scale_max_ref_point=False
    )

    # Run error analysis
    std_devs = ehvi_error_analysis(
        query_smiles=query_smiles[:1],  # Evaluate error on the first query point for simplicity
        known_smiles=training_smiles,
        known_Y=training_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        reference_point=reference_point,
        N_values=N_values,
        num_runs=num_runs
    )

    # Plotting the standard deviation vs N
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, std_devs, marker='o', color='blue', label='EHVI Standard Deviation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Standard Deviation of EHVI Estimate')
    plt.title('Variability of EHVI Estimate with Increasing Monte Carlo Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_error_analysis_experiment()
