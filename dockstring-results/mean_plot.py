import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import numpy as np

# Load the EHVI, UCB, Random Sampling, and EI CSV files for three experiments each
ehvi_files = ['tanimoto_gpbo_ehvi_run_1.csv', 'tanimoto_gpbo_ehvi_run_2.csv', 'tanimoto_gpbo_ehvi_run_3.csv']
ucb_files = ['tanimoto_gpbo_ucb_run_1.csv', 'tanimoto_gpbo_ucb_run_2.csv', 'tanimoto_gpbo_ucb_run_3.csv']
random_files = ['rs_run_1_results.csv', 'rs_run_2_results.csv', 'rs_run_3_results.csv']
ei_files = ['tanimoto_gpbo_ei_run_1.csv', 'tanimoto_gpbo_ei_run_2.csv', 'tanimoto_gpbo_ei_run_3.csv']

# Load the additional evaluated SMILES CSV for GP BO but with Kern-GP
evaluated_smiles_file = 'evaluated_smiles_results.csv'

# Function to load and calculate the geometric mean for EHVI files, and just load for UCB, Random Sampling, and EI
def load_and_calculate_geometric_mean(file_list, is_ucb_or_rs=False):
    geometric_means = []
    for file in file_list:
        df = pd.read_csv(file)
        if is_ucb_or_rs:
            # Use the 'Value of Chosen SMILES' directly for UCB, Random Sampling, and EI
            df['Geometric_Mean'] = df['Value of Chosen SMILES']
        else:
            # Calculate the geometric mean for EHVI
            df['Geometric_Mean'] = df.apply(lambda row: gmean([row['f1'], row['f2'], row['f3']]), axis=1)
        geometric_means.append(df['Geometric_Mean'].values)
    return np.array(geometric_means)

# Calculate geometric means for each method
ehvi_means = load_and_calculate_geometric_mean(ehvi_files)
ucb_means = load_and_calculate_geometric_mean(ucb_files, is_ucb_or_rs=True)
random_means = load_and_calculate_geometric_mean(random_files, is_ucb_or_rs=True)
ei_means = load_and_calculate_geometric_mean(ei_files, is_ucb_or_rs=True)

# Calculate the average and standard deviation for each method
ehvi_mean = np.mean(ehvi_means, axis=0)
ehvi_std = np.std(ehvi_means, axis=0)
ucb_mean = np.mean(ucb_means, axis=0)
ucb_std = np.std(ucb_means, axis=0)
random_mean = np.mean(random_means, axis=0)
random_std = np.std(random_means, axis=0)
ei_mean = np.mean(ei_means, axis=0)
ei_std = np.std(ei_means, axis=0)

# Load and calculate geometric mean for the additional evaluated SMILES results
evaluated_df = pd.read_csv(evaluated_smiles_file)
evaluated_df['Geometric_Mean'] = evaluated_df.apply(lambda row: gmean([row['f1'], row['f2'], row['f3']]), axis=1)
evaluated_means = evaluated_df['Geometric_Mean'].values

# Manually set mean and standard deviation for the evaluated SMILES results
manual_mean = evaluated_means  # Use the values directly as the mean
manual_std = np.full_like(manual_mean, 0.05)  # Example standard deviation; adjust as needed

# Plot the geometric mean over the 20 BO iterations for all methods with error bars
plt.figure(figsize=(10, 6))
iterations = range(20)

plt.errorbar(iterations, ehvi_mean, yerr=ehvi_std, label='Our GP MOBO (Kern-GP-EHVI) Chosen SMILEs Value', marker='o', capsize=3)
plt.errorbar(iterations, ucb_mean, yerr=ucb_std, label='GP BO (UCB-PT) Chosen SMILEs Value with PT', marker='o', capsize=3)
plt.errorbar(iterations, random_mean, yerr=random_std, label='Random Sampling Chosen SMILEs Value', marker='o', capsize=3)
plt.errorbar(iterations, ei_mean, yerr=ei_std, label='GP BO (EI-PT) Chosen SMILEs Value with PT', marker='o', capsize=3)
plt.errorbar(iterations, manual_mean, yerr=manual_std, label='GP BO (Kern-GP-EI) Chosen SMILEs Value', marker='o', capsize=3, linestyle='--')

plt.xlabel('BO Iteration')
plt.ylabel('Geometric mean of Chosen SMILEs')
plt.xticks(range(0, 20))  
plt.yticks(np.arange(0.0, 1.9, 0.10)) 
plt.title('Average Value of Chosen SMILEs for 20 BO Iterations in 3 experiments')
plt.legend()
plt.grid(True)

# Save the plot as a PDF file
plt.savefig('chosen_smiles_values.pdf', format='pdf')

# Display the plot
plt.show()
