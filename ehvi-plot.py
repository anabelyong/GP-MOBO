import matplotlib.pyplot as plt
import numpy as np
from dockstring.dataset import load_dataset
from scipy.stats import norm

from acquisition_funcs.hypervolume import compute_hypervolume, infer_reference_point
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives


def expected_hypervolume_improvement(pred_means, pred_vars, reference_point):
    num_points, num_objectives = pred_means.shape
    ehvi_values = np.zeros(num_points)

    for i in range(num_points):
        mean = pred_means[i]
        var = pred_vars[i]
        std = np.sqrt(var)

        hvi = 0.0
        for j in range(num_objectives):
            u = (reference_point[j] - mean[j]) / std[j]
            cdf_u = norm.cdf(u)
            pdf_u = norm.pdf(u)
            hvi += (reference_point[j] - mean[j]) * cdf_u + std[j] * pdf_u

        ehvi_values[i] = hvi

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

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point)

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

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        # Infer the reference point dynamically based on the Pareto front points
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

        current_hypervolume = compute_hypervolume(known_Y, reference_point)
        hypervolumes_bo.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

    return known_smiles, known_Y, hypervolumes_bo


def random_evaluation_loop(
    known_smiles, query_smiles, known_Y, max_ref_point=None, scale=0.1, scale_max_ref_point=False, n_iterations=20
):
    hypervolumes_random = []

    for iteration in range(n_iterations):
        print(f"Start Random iteration {iteration}. Dataset size={known_Y.shape}")

        random_smiles = np.random.choice(query_smiles, 1)[0]
        new_Y = evaluate_objectives([random_smiles])
        known_smiles.append(random_smiles)
        known_Y = np.vstack([known_Y, new_Y])
        print(f"Chosen SMILES: {random_smiles}")
        print(f"Value of chosen SMILES: {new_Y}")
        print(f"Updated dataset size: {known_Y.shape}")

        # Infer the reference point dynamically based on the Pareto front points
        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        current_hypervolume = compute_hypervolume(known_Y, reference_point)
        hypervolumes_random.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

    return known_smiles, known_Y, hypervolumes_random


# Example usage:
if __name__ == "__main__":
    # Load dockstring dataset
    DOCKSTRING_DATASET = load_dataset()
    ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
    initial_smiles = ALL_SMILES[:20]
    known_smiles = initial_smiles.copy()
    query_smiles = ALL_SMILES[20:10_000]

    # Evaluate objectives for known smiles
    known_Y = evaluate_objectives(known_smiles)

    # Define GP hyperparameters
    gp_means = np.asarray([0.0, 0.0, 1.0, 0.0])
    gp_amplitudes = np.asarray([1.0, 0.5, 1.0, 0.5])
    gp_noises = np.asarray([1.0, 1e-4, 1e-1, 1e-2])

    # Perform Bayesian optimization loop
    known_smiles_bo, known_Y_bo, hypervolumes_bo = bayesian_optimization_loop(
        known_smiles=known_smiles.copy(),
        query_smiles=query_smiles,
        known_Y=known_Y.copy(),
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=20,
    )

    # Perform random evaluation loop
    known_smiles_random, known_Y_random, hypervolumes_random = random_evaluation_loop(
        known_smiles=known_smiles.copy(), query_smiles=query_smiles, known_Y=known_Y.copy(), n_iterations=20
    )

    # Plot hypervolume over iterations for both methods
    plt.plot(range(len(hypervolumes_bo)), hypervolumes_bo, label="Bayesian Optimization", color="blue")
    plt.fill_between(
        range(len(hypervolumes_bo)),
        np.array(hypervolumes_bo) - np.std(hypervolumes_bo),
        np.array(hypervolumes_bo) + np.std(hypervolumes_bo),
        alpha=0.3,
        color="blue",
    )

    plt.plot(range(len(hypervolumes_random)), hypervolumes_random, label="Random Evaluation", color="red")
    plt.fill_between(
        range(len(hypervolumes_random)),
        np.array(hypervolumes_random) - np.std(hypervolumes_random),
        np.array(hypervolumes_random) + np.std(hypervolumes_random),
        alpha=0.3,
        color="red",
    )

    plt.xlabel("Evaluation point number")
    plt.ylabel("Dominated Hypervolume")
    plt.title("Dominated Hypervolume over Evaluation Points")
    plt.legend()
    plt.show()
