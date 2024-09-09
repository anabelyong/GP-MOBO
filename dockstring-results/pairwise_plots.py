import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from tdc import Oracle
from dockstring.dataset import load_dataset

# Function to load the datasets for each experiment
def load_experiment_data(ehvi_file, ucb_file, pareto_file):
    ehvi_df = pd.read_csv(ehvi_file)
    ucb_df = pd.read_csv(ucb_file)
    pareto_df = pd.read_csv(pareto_file)
    return ehvi_df, ucb_df, pareto_df

# Function to evaluate objectives
def evaluate_objectives(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array([-DOCKSTRING_DATASET["PPARD"].get(s, np.nan) for s in smiles_list])
    f2 = np.array(QED_ORACLE(smiles_list))
    f3 = np.array(CELECOXIB_ORACLE(smiles_list))

    # Replace NaNs with a small value to avoid numerical instability
    f1 = np.nan_to_num(f1, nan=1e-6)
    f2 = np.nan_to_num(f2, nan=1e-6)
    f3 = np.nan_to_num(f3, nan=1e-6)

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3])
    return out.T

# Function to plot pairwise comparisons for each experiment
def plot_experiment(axs, pareto_df, ehvi_df, ucb_df, experiment_number):
    # f1 vs f2
    axs[0].scatter(ehvi_df['f1'], ehvi_df['f2'], label='Our GP-MOBO (KERN-GP-EHVI)', color='blue')
    axs[0].scatter(ucb_df['f1'], ucb_df['f2'], label='GP-BO (UCB-PT)', color='orange')
    axs[0].scatter(pareto_df['f1'], pareto_df['f2'], label='Pareto Points', color='red')
    axs[0].set_xlabel('f1')
    axs[0].set_ylabel('f2')
    axs[0].set_title(f'Experiment {experiment_number} - f1 vs f2')

    # f1 vs f3
    axs[1].scatter(ehvi_df['f1'], ehvi_df['f3'], label='Our GP-MOBO (KERN-GP-EHVI)', color='blue')
    axs[1].scatter(ucb_df['f1'], ucb_df['f3'], label='GP-BO (UCB-PT)', color='orange')
    axs[1].scatter(pareto_df['f1'], pareto_df['f3'], label='Pareto Points', color='red')
    axs[1].set_xlabel('f1')
    axs[1].set_ylabel('f3')
    axs[1].set_title(f'Experiment {experiment_number} - f1 vs f3')

    # f2 vs f3
    axs[2].scatter(ehvi_df['f2'], ehvi_df['f3'], label='Our GP-MOBO (KERN-GP-EHVI)', color='blue')
    axs[2].scatter(ucb_df['f2'], ucb_df['f3'], label='GP-BO (UCB-PT)', color='orange')
    axs[2].scatter(pareto_df['f2'], pareto_df['f3'], label='Pareto Points', color='red')
    axs[2].set_xlabel('f2')
    axs[2].set_ylabel('f3')
    axs[2].set_title(f'Experiment {experiment_number} - f2 vs f3')

# Load the dockstring dataset and oracles
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")

# Define file paths for each experiment
ehvi_files = [
    'tanimoto_gpbo_ehvi_run_1.csv',
    'tanimoto_gpbo_ehvi_run_2.csv',
    'tanimoto_gpbo_ehvi_run_3.csv'
]
ucb_files = [
    'tanimoto_gpbo_ucb_run_1.csv',
    'tanimoto_gpbo_ucb_run_2.csv',
    'tanimoto_gpbo_ucb_run_3.csv'
]
pareto_file = 'pareto_points.csv'  # Assuming pareto points file is the same for all experiments

# Create a 3x3 grid for the subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, (ehvi_file, ucb_file) in enumerate(zip(ehvi_files, ucb_files), start=1):
    ehvi_df, ucb_df, pareto_df = load_experiment_data(ehvi_file, ucb_file, pareto_file)

    # Evaluate UCB objectives for current experiment
    ucb_smiles = ucb_df['Chosen SMILES'].tolist()
    ucb_evaluated = evaluate_objectives(ucb_smiles)
    ucb_df['f1'], ucb_df['f2'], ucb_df['f3'] = ucb_evaluated.T

    # Plot the pairwise comparisons for this experiment
    plot_experiment(axs[i-1, :], pareto_df, ehvi_df, ucb_df, experiment_number=i)

# Set layout and save the figure as a PDF
plt.tight_layout()
plt.savefig('combined_experiments_pairwise_plots.pdf', format='pdf')

# Show the plot
plt.show()
