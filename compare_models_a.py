import pandas as pd
import numpy as np
from scipy.stats import gmean, wilcoxon

# Load the CSV files
ehvi_df = pd.read_csv('/Users/anabelyong/kernel-mobo/results_comparison/ehvi_bo_fexofenadine_results_2.csv')
ei_df = pd.read_csv('/Users/anabelyong/kernel-mobo/ei_bo_rs_experiments_fexofenadine_results.csv')

# Calculate the geometric means for the EHVI results
ehvi_df['Geometric_Mean_EHVI_MPO'] = ehvi_df[['EHVI-MPO-F1', 'EHVI-MPO-F2', 'EHVI-MPO-F3']].apply(gmean, axis=1)

# Reshape the data so that we have each experiment's geometric means for 20 iterations
ehvi_gmeans = ehvi_df.pivot_table(index='BO Iteration', columns='Experiment', values='Geometric_Mean_EHVI_MPO')
ei_gmeans = ei_df.pivot_table(index='BO Iteration', columns='Experiment', values='Chosen Smiles for Fex-MPO')

# Compute the mean across the three experiments for each BO iteration
mean_ehvi_gmeans = ehvi_gmeans.mean(axis=1)
mean_ei_gmeans = ei_gmeans.mean(axis=1)

# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(mean_ehvi_gmeans, mean_ei_gmeans)

# Output the results
print(f"Wilcoxon signed-rank test statistic: {stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The test is significant at p < 0.05, suggesting that the EHVI method is statistically better than the EI method.")
else:
    print("The test is not significant at p < 0.05, suggesting no significant difference between EHVI and EI methods.")
