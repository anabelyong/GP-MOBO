import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

df = pd.read_csv('plot2.csv')

# Ensure column names are correct
print(df.columns)
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces

# Calculate the geometric mean of EHVI-MPO-F1, EHVI-MPO-F2, and EHVI-MPO-F3
df['Geometric_Mean_EHVI_MPO'] = df[['EHVI-MPO-F1', 'EHVI-MPO-F2', 'EHVI-MPO-F3']].apply(gmean, axis=1)

# Plotting the EI MPO and Geometric Mean of EHVI MPO over iterations
plt.figure(figsize=(10, 6))

# EI MPO values
plt.plot(df['BO Iteration'], df['EI MPO'], marker='o', label='EI MPO', color='blue')

# Geometric Mean of EHVI MPO values
plt.plot(df['BO Iteration'], df['Geometric_Mean_EHVI_MPO'], marker='o', label='Geometric Mean of EHVI MPO', color='red')

# Adding labels and title
plt.xlabel('BO Iteration')
plt.ylabel('Best SMILES Values')
plt.title('Comparison of EI MPO and Geometric Mean of EHVI MPO over BO Iterations')
plt.legend()

# Set custom tick labels for the x-axis to show iterations 1-20
plt.xticks(ticks=np.arange(1, 21), labels=[str(i) for i in range(1, 21)])

# Display the plot
plt.grid(True)
plt.show()