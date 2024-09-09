import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import numpy as np

# Load the EHVI, EI, and Random Sampling CSV files
ehvi_df = pd.read_csv('tanimoto_gpbo_ehvi_run_3.csv')
ei_df = pd.read_csv('tanimoto_gpbo_ucb_run_3.csv')
random_df = pd.read_csv('rs_results_experiment_1.csv')  # Replace with your actual file name

# Calculate the geometric mean for each iteration in the EHVI, EI, and Random Sampling data
ehvi_df['Geometric_Mean'] = ehvi_df.apply(lambda row: gmean([row['f1'], row['f2'], row['f3']]), axis=1)
ei_df['Geometric_Mean'] = ei_df['Value of Chosen SMILES']  # Assuming 'Chosen SMILES' already has the geometric mean
random_df['Geometric_Mean'] = random_df.apply(lambda row: gmean([row['f1'], row['f2'], row['f3']]), axis=1)

# Plot the geometric mean over the 20 BO iterations for EHVI, EI, and Random Sampling
plt.figure(figsize=(10, 6))
plt.plot(ehvi_df['Iteration'], ehvi_df['Geometric_Mean'], label='GP MOBO (EHVI) Chosen SMILEs Value', marker='o')
plt.plot(ei_df['Iteration'], ei_df['Geometric_Mean'], label='GP BO (UCB) Chosen SMILEs Value', marker='o')
plt.plot(random_df['Iteration'], random_df['Geometric_Mean'], label='Random Sampling Chosen SMILEs Value', marker='o')

plt.xlabel('BO Iteration')
plt.ylabel('Geometric Mean')
plt.xticks(range(0, 21))  
plt.yticks(np.arange(0.0, 1.5, 0.10)) 
plt.title('Value of Chosen SMILEs over 20 BO Iterations')
plt.legend()
plt.grid(True)
plt.show()
