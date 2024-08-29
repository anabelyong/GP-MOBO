import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import gmean

# Load the CSV files
ei_df = pd.read_csv('evaluated_smiles_results.csv')
ehvi_df = pd.read_csv('/Users/anabelyong/kernel-mobo/results/bo_results_1.csv')

# Calculate the geometric mean for f1, f2, and f3
ei_df['geometric_mean'] = ei_df[['f1', 'f2', 'f3']].apply(gmean, axis=1)
ehvi_df['geometric_mean'] = ehvi_df[['f1', 'f2', 'f3']].apply(gmean, axis=1)

# Perform the Wilcoxon signed-rank test on the geometric means
stat, p_value = wilcoxon(ei_df['geometric_mean'], ehvi_df['geometric_mean'])

print(f"Wilcoxon signed-rank test for geometric means:")
print(f"  Statistic: {stat}")
print(f"  P-value: {p_value}")

if p_value < 0.05:
    print(f"  The EHVI model significantly outperforms the EI model based on the geometric mean.")
else:
    print(f"  No significant difference between the EHVI and EI models based on the geometric mean.")

