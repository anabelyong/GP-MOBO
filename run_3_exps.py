import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from dockstring.dataset import load_dataset
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives


def expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_front, N=100):
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

    return ehvi_values


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

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        max_acq = -np.inf
        best_smiles = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ehvi_values = ehvi_acquisition(
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

        acquisition_values.append(max_acq)
        print(f"Max acquisition value: {max_acq}")
        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_objectives([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            print(f"Chosen SMILES: {best_smiles} with acquisition function value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
        else:
            print("No new SMILES selected.")

        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_bo.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

        # Append results to list for CSV writing
        results.append([iteration, max_acq, best_smiles, new_Y[0][0], new_Y[0][1], new_Y[0][2], current_hypervolume])

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values, results


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
        new_Y = evaluate_objectives([best_smiles])
        known_smiles.append(best_smiles)
        known_Y = np.vstack([known_Y, new_Y])
        print(f"Chosen SMILES: {best_smiles}")
        print(f"Value of chosen SMILES: {new_Y}")
        print(f"Updated dataset size: {known_Y.shape}")

        reference_point = infer_reference_point(known_Y)

        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_rs.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

        acquisition_values_rs.append(0)  # Random sampling does not use acquisition values
        results_rs.append([iteration, 0, best_smiles, new_Y[0][0], new_Y[0][1], new_Y[0][2], current_hypervolume])

    return known_smiles, known_Y, hypervolumes_rs, acquisition_values_rs, results_rs


def plot_comparison(hypervolumes_bo, acquisition_values_bo, hypervolumes_rs, acquisition_values_rs):
    iterations = np.arange(1, len(hypervolumes_bo) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Hypervolume", color="tab:blue")
    ax1.plot(iterations, hypervolumes_bo, "o-", color="tab:blue", label="Hypervolume BO")
    ax1.plot(iterations, hypervolumes_rs, "x-", color="tab:cyan", label="Hypervolume RS")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Acquisition Function Value", color="tab:red")
    ax2.plot(iterations, acquisition_values_bo, "o-", color="tab:red", label="Acquisition Value BO")
    ax2.plot(iterations, acquisition_values_rs, "x-", color="tab:orange", label="Acquisition Value RS")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Hypervolume and Acquisition Function Value over Iterations")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.show()


def plot_3d_pareto_and_samples(known_Y_bo, pareto_Y_bo, known_Y_rs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(known_Y_bo[:, 0], known_Y_bo[:, 1], known_Y_bo[:, 2], c="blue", label="BO Samples")
    ax.scatter(known_Y_rs[:, 0], known_Y_rs[:, 1], known_Y_rs[:, 2], c="green", label="RS Samples")
    ax.scatter(pareto_Y_bo[:, 0], pareto_Y_bo[:, 1], pareto_Y_bo[:, 2], c="red", label="Pareto Optimal Front")

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.legend()
    plt.show()


def write_to_csv(results, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Acquisition Function Value", "Chosen SMILES", "f1", "f2", "f3", "Hypervolume"])
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    DOCKSTRING_DATASET = load_dataset()
    ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]

    for experiment in range(1, 4):
        print(f"Running Experiment {experiment}")
        initial_smiles = random.sample(ALL_SMILES, 20)
        known_smiles = initial_smiles.copy()
        query_smiles = [smiles for smiles in ALL_SMILES if smiles not in known_smiles]

        known_Y = evaluate_objectives(known_smiles)

        gp_means = np.asarray([0.0, 0.0, 1.0])
        gp_amplitudes = np.asarray([1.0, 0.5, 1.0])
        gp_noises = np.asarray([1.0, 1e-4, 1e-1])

        # Bayesian Optimization Loop
        known_smiles_bo, known_Y_bo, hypervolumes_bo, acquisition_values_bo, results_bo = bayesian_optimization_loop(
            known_smiles=known_smiles.copy(),
            query_smiles=query_smiles.copy(),
            known_Y=known_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=20,
        )

        # Random Sampling Loop
        known_smiles_rs, known_Y_rs, hypervolumes_rs, acquisition_values_rs, results_rs = random_sampling_loop(
            known_smiles=known_smiles.copy(),
            query_smiles=query_smiles.copy(),
            known_Y=known_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=20,
        )

        # Pareto Fronts
        pareto_mask_bo = pareto_front(known_Y_bo)
        pareto_Y_bo = known_Y_bo[pareto_mask_bo]

        # Plotting
        plot_3d_pareto_and_samples(known_Y_bo, pareto_Y_bo, known_Y_rs)
        plot_comparison(hypervolumes_bo, acquisition_values_bo, hypervolumes_rs, acquisition_values_rs)

        # Write results to CSV files
        write_to_csv(results_bo, f"bo_results_experiment_{experiment}.csv")
        write_to_csv(results_rs, f"rs_results_experiment_{experiment}.csv")
