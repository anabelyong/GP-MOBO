import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Load the first CSV file (EHVI results)
df_ehvi = pd.read_csv('ehvi_bo_fexofenadine_results.csv')

# Load the second CSV file (EI results)
df_ei_rs = pd.read_csv('ei_bo_rs_experiments_fexofenadine_results.csv')

# Ensure column names are correct
df_ehvi.columns = df_ehvi.columns.str.strip()  # Strip any leading/trailing spaces
df_ei_rs.columns = df_ei_rs.columns.str.strip()  # Strip any leading/trailing spaces

# Calculate the geometric mean of EHVI-MPO-F1, EHVI-MPO-F2, and EHVI-MPO-F3 for each row
df_ehvi['Geometric_Mean_EHVI_MPO'] = df_ehvi[['EHVI-MPO-F1', 'EHVI-MPO-F2', 'EHVI-MPO-F3']].apply(gmean, axis=1)

# Group by 'BO Iteration' and calculate the mean of the geometric means across all experiments
ehvi_mean = df_ehvi.groupby('BO Iteration')['Geometric_Mean_EHVI_MPO'].mean().reset_index()

# Group by 'BO Iteration' and calculate the mean for 'Chosen Smiles for Fex-MPO' and 'RS-MPO'
ei_rs_mean = df_ei_rs.groupby('BO Iteration')[['Chosen Smiles for Fex-MPO', 'RS-MPO']].mean().reset_index()

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the mean geometric mean across the experiments (EHVI MPO)
plt.plot(ehvi_mean['BO Iteration'], ehvi_mean['Geometric_Mean_EHVI_MPO'], 
         marker='o', label='Mean Geometric Mean of EHVI MPO (Across 3 Experiments)', color='blue', linestyle='-')

# Plot the mean of 'Chosen Smiles for Fex-MPO' (EI results)
plt.plot(ei_rs_mean['BO Iteration'], ei_rs_mean['Chosen Smiles for Fex-MPO'], 
         marker='s', label='Mean Chosen Smiles for Fex-MPO (EI)', color='green', linestyle='--')

# Plot the mean of 'RS-MPO' (Random Sampling MPO)
plt.plot(ei_rs_mean['BO Iteration'], ei_rs_mean['RS-MPO'], 
         marker='^', label='Mean RS-MPO (Random Sampling)', color='orange', linestyle='-.')

# Adding labels and title
plt.xlabel('BO Iteration')
plt.ylabel('Geometric Mean of Best Chosen SMILEs')
plt.title('Comparison of EHVI vs EI in tackling Fexofenadine MPO')
plt.legend()

# Set custom tick labels for the x-axis to show iterations 1-20
plt.xticks(ticks=np.arange(1, 21), labels=[str(i) for i in range(1, 21)])

# Display the plot
plt.grid(True)
plt.show()
