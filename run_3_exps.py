import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from dockstring.dataset import load_dataset
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils import evaluate_objectives
from ehvi_sampling import expected_hypervolume_improvement, ehvi_acquisition, sample_from_gp_posterior, momcmc_sampling

def momcmc_sampling(known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, num_samples, temperature, desired_acceptance, reference_point, N=100):
    P = known_Y.copy()
    N_points = P.shape[0]
    acceptance_rate = 0.0

    hv = Hypervolume(reference_point)

    while acceptance_rate < desired_acceptance:
        P_new = np.empty_like(P)
        samples = sample_from_gp_posterior(known_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, N)

        for i in range(N_points):
            # Select a random sample
            sample_idx = np.random.randint(N)
            v = samples[sample_idx, i, :]

            if pareto_front(np.vstack([P, v]))[-1]:
                fit_new = hv.compute(np.vstack([P, v]))
                fit_old = hv.compute(P)
                if np.random.rand() < np.exp((fit_old - fit_new) / temperature):
                    P_new[i] = v
                else:
                    P_new[i] = P[i]
            else:
                P_new[i] = P[i]
        acceptance_rate = np.mean(np.all(P_new == P, axis=1))
        P = P_new
        if acceptance_rate < desired_acceptance:
            temperature *= 1.1
        else:
            temperature *= 0.9
    return P

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

def run_experiment(n_runs=3):
    bo_results = []
    rs_results = []

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")

        DOCKSTRING_DATASET = load_dataset()
        ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
        
        # Introduce randomness
        random.shuffle(ALL_SMILES)
        
        initial_smiles = ALL_SMILES[:20]
        known_smiles = initial_smiles.copy()
        query_smiles = ALL_SMILES[20:10_000]

        known_Y = evaluate_objectives(known_smiles)

        gp_means = np.asarray([0.0, 0.0, 1.0])
        gp_amplitudes = np.asarray([1.0, 0.5, 1.0])
        gp_noises = np.asarray([1.0, 1e-4, 1e-1])

        known_smiles_bo, known_Y_bo, hypervolumes_bo, acquisition_values_bo = bayesian_optimization_loop(
            known_smiles=known_smiles.copy(),
            query_smiles=query_smiles.copy(),
            known_Y=known_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=20,
        )

        known_smiles_rs, known_Y_rs, hypervolumes_rs, acquisition_values_rs = random_sampling_loop(
            known_smiles=known_smiles.copy(),
            query_smiles=query_smiles.copy(),
            known_Y=known_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=20,
        )

        # Save results for each run
        bo_results.append((hypervolumes_bo, acquisition_values_bo))
        rs_results.append((hypervolumes_rs, acquisition_values_rs))

    return bo_results, rs_results

def save_results_to_csv(results, filename):
    rows = []
    for run_index, (hypervolumes, acquisition_values) in enumerate(results):
        for iteration, (hv, acq) in enumerate(zip(hypervolumes, acquisition_values)):
            rows.append([run_index + 1, iteration + 1, hv, acq])
    df = pd.DataFrame(rows, columns=["Run", "Iteration", "Hypervolume", "Acquisition Value"])
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    bo_results, rs_results = run_experiment(n_runs=3)

    save_results_to_csv(bo_results, "bo_results.csv")
    save_results_to_csv(rs_results, "rs_results.csv")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    # Plot results from one of the runs for illustration
    hypervolumes_bo, acquisition_values_bo = bo_results[0]
    hypervolumes_rs, acquisition_values_rs = rs_results[0]

    plot_comparison(hypervolumes_bo, hypervolumes_rs, acquisition_values_bo, acquisition_values_rs)

    
