#correct MC integration method implementation
import numpy as np
import matplotlib.pyplot as plt
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

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values

def plot_results(hypervolumes, acquisition_values):
    iterations = np.arange(1, len(hypervolumes) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Hypervolume', color='tab:blue')
    ax1.plot(iterations, hypervolumes, 'o-', color='tab:blue', label='Hypervolume')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Acquisition Function Value', color='tab:red')
    ax2.plot(iterations, acquisition_values, 'o-', color='tab:red', label='Acquisition Function Value')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Hypervolume and Acquisition Function Value over BO Iterations')
    plt.show()

def plot_3d_pareto_and_samples(known_Y, pareto_Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(known_Y[:, 0], known_Y[:, 1], known_Y[:, 2], c="blue", label="Samples")
    ax.scatter(pareto_Y[:, 0], pareto_Y[:, 1], pareto_Y[:, 2], c="red", label="Pareto Optimal Front")

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    DOCKSTRING_DATASET = load_dataset()
    ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
    initial_smiles = ALL_SMILES[:20]
    known_smiles = initial_smiles.copy()
    query_smiles = ALL_SMILES[20:10_000]

    known_Y = evaluate_objectives(known_smiles)

    gp_means = np.asarray([0.0, 0.0, 1.0])
    gp_amplitudes = np.asarray([1.0, 0.5, 1.0])
    gp_noises = np.asarray([1.0, 1e-4, 1e-1])

    known_smiles_bo, known_Y_bo, hypervolumes_bo, acquisition_values_bo = bayesian_optimization_loop(
        known_smiles=known_smiles.copy(),
        query_smiles=query_smiles,
        known_Y=known_Y.copy(),
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=20,
    )

    pareto_mask = pareto_front(known_Y_bo)
    pareto_Y_bo = known_Y_bo[pareto_mask]

    plot_3d_pareto_and_samples(known_Y_bo, pareto_Y_bo)
    plot_results(hypervolumes_bo, acquisition_values_bo)
