import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from dockstring.dataset import load_dataset
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives
from ehvi_sampling import expected_hypervolume_improvement, ehvi_acquisition, sample_from_gp_posterior, momcmc_sampling

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
    acquisition_values_bo = []
    temperature = 1.0
    desired_acceptance = 0.1

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

        acquisition_values_bo.append(max_acq)
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

        known_Y = momcmc_sampling(
            known_smiles=known_smiles,
            known_Y=known_Y,
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            num_samples=known_Y.shape[0],
            temperature=temperature,
            desired_acceptance=desired_acceptance,
            reference_point=reference_point,
            N=100
        )

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values_bo

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

    for iteration in range(n_iterations):
        print(f"Start Random Sampling iteration {iteration}. Dataset size={known_Y.shape}")

        # Randomly sample a query_smile
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

        # Compute EHVI for comparison
        pred_means, pred_vars = independent_tanimoto_gp_predict(
            query_smiles=[best_smiles],
            known_smiles=known_smiles,
            known_Y=known_Y,
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
        )
        pareto_mask = pareto_front(known_Y)
        pareto_Y = known_Y[pareto_mask]
        ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y)
        acquisition_values_rs.append(ehvi_values[0])

    return known_smiles, known_Y, hypervolumes_rs, acquisition_values_rs

def plot_comparison(hypervolumes_bo, hypervolumes_rs, acquisition_values_bo, acquisition_values_rs):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(hypervolumes_bo, label="Bayesian Optimization (EHVI)")
    plt.plot(hypervolumes_rs, label="Random Sampling")
    plt.xlabel("Iterations")
    plt.ylabel("Hypervolume")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acquisition_values_bo, label="Bayesian Optimization (EHVI)")
    plt.plot(acquisition_values_rs, label="Random Sampling")
    plt.xlabel("Iterations")
    plt.ylabel("Acquisition Function Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_3d_pareto_and_samples(known_Y, pareto_Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(known_Y[:, 0], known_Y[:, 1], known_Y[:, 2], c="blue", label="MOMCMC")
    ax.scatter(pareto_Y[:, 0], pareto_Y[:, 1], pareto_Y[:, 2], c="red", label="Pareto Optimal Front")

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

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

    known_smiles_rs, known_Y_rs, hypervolumes_rs, acquisition_values_rs = random_sampling_loop(
        known_smiles=known_smiles.copy(),
        query_smiles=query_smiles,
        known_Y=known_Y.copy(),
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=20,
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    pareto_mask_bo = pareto_front(known_Y_bo)
    pareto_Y_bo = known_Y_bo[pareto_mask_bo]

    pareto_mask_rs = pareto_front(known_Y_rs)
    pareto_Y_rs = known_Y_rs[pareto_mask_rs]

    plot_comparison(hypervolumes_bo, hypervolumes_rs, acquisition_values_bo, acquisition_values_rs)
    plot_3d_pareto_and_samples(known_Y_bo, pareto_Y_bo)
    plot_3d_pareto_and_samples(known_Y_rs, pareto_Y_rs)
