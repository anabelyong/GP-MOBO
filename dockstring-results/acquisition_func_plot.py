import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load the CSV files and calculate mean and standard deviation for Acquisition Function Value and Hypervolume
def calculate_mean_std_acquisition(file_list):
    acquisition_values = []
    hypervolume_values = []
    
    for file in file_list:
        df = pd.read_csv(file)
        # Extract Acquisition Function Value and Hypervolume columns
        acquisition_values.append(df['Acquisition Function Value'].values)
        hypervolume_values.append(df['Hypervolume'].values)
    
    # Convert to numpy arrays for easy manipulation
    acquisition_values = np.array(acquisition_values)
    hypervolume_values = np.array(hypervolume_values)
    
    # Calculate the mean and standard deviation over the 20 BO iterations
    acquisition_mean = np.mean(acquisition_values, axis=0)
    acquisition_std = np.std(acquisition_values, axis=0)
    
    hypervolume_mean = np.mean(hypervolume_values, axis=0)
    hypervolume_std = np.std(hypervolume_values, axis=0)
    
    return acquisition_mean, acquisition_std, hypervolume_mean, hypervolume_std

# List of your CSV files
csv_files = ['tanimoto_gpbo_ehvi_run_1.csv', 'tanimoto_gpbo_ehvi_run_2.csv', 'tanimoto_gpbo_ehvi_run_3.csv']

# Calculate mean and std for Acquisition Function Value and Hypervolume
acquisition_mean, acquisition_std, hypervolume_mean, hypervolume_std = calculate_mean_std_acquisition(csv_files)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

iterations = range(20)  # Assuming 20 iterations

# Plotting the results for Acquisition Function Value
axs[0].errorbar(iterations, acquisition_mean, yerr=acquisition_std, label='Acquisition Function Mean', marker='o', capsize=3)
axs[0].set_xlabel('BO Iteration')
axs[0].set_ylabel('Acquisition Function (EHVI Value)')
axs[0].set_title('Average Acquisition Function Value over 20 BO Iterations for 3 experiments')
axs[0].set_xticks(range(0, 21))  
axs[0].grid(True)
axs[0].legend()

# Plotting the results for Hypervolume
axs[1].errorbar(iterations, hypervolume_mean, yerr=hypervolume_std, label='Hypervolume Mean', marker='o', capsize=3)
axs[1].set_xlabel('BO Iteration')
axs[1].set_ylabel('Hypervolume (HV)')
axs[1].set_title('Average Hypervolume over 20 BO Iterations for 3 experiments')
axs[1].set_xticks(range(0, 21))  
axs[1].grid(True)
axs[1].legend()

# Adjust layout and save the figure as an image (e.g., PNG or PDF)
plt.tight_layout()
plt.savefig('acquisition_hypervolume_plots.pdf', format='pdf')

# Show the plot
plt.show()
