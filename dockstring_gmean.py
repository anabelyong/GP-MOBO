import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

def calculate_geometric_mean(df, columns):
    """
    Calculate the geometric mean for the specified columns in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    columns (list of str): The list of columns to calculate the geometric mean over.
    
    Returns:
    np.ndarray: Array of geometric means.
    """
    data = df[columns].values
    geo_means = gmean(data, axis=1)
    return geo_means

def plot_geometric_means(iterations, bo_geo_means, rs_geo_means, random_geo_means):
    """
    Plot the geometric means over iterations for EHVI MPO, EI MPO, and Random Sampling.
    
    Parameters:
    iterations (list of int): List of iteration numbers.
    bo_geo_means (np.ndarray): Array of geometric means from BO results.
    rs_geo_means (np.ndarray): Array of geometric means from RS results.
    random_geo_means (np.ndarray): Array of geometric means from Random Sampling results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, bo_geo_means, label="EHVI MPO Geometric Mean", marker='o', color='blue')
    plt.plot(iterations, rs_geo_means, label="EI MPO Geometric Mean", marker='x', color='orange')
    plt.plot(iterations, random_geo_means, label="Random Sampling Geometric Mean", marker='s', color='green')
    plt.xlabel('BO Iteration')
    plt.ylabel('Geometric Mean')
    plt.title('Geometric Mean of f1, f2, f3 over Iterations')
    plt.xticks(ticks=iterations)  # Set x-axis ticks to integer values 1 to 20
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Read the CSV files for EHVI MPO, EI MPO, and Random Sampling
    bo_df = pd.read_csv('bo_results.csv')
    rs_df = pd.read_csv('evaluated_smiles_results.csv')
    random_df = pd.read_csv('/Users/anabelyong/kernel-mobo/results/rs_results.csv')  # Add the random sampling CSV file here
    
    # Calculate geometric means for all three methods
    bo_geo_means = calculate_geometric_mean(bo_df, ['f1', 'f2', 'f3'])
    rs_geo_means = calculate_geometric_mean(rs_df, ['f1', 'f2', 'f3'])
    random_geo_means = calculate_geometric_mean(random_df, ['f1', 'f2', 'f3'])
    
    iterations = np.arange(1, len(bo_geo_means) + 1)
    
    # Plot all three sets of geometric means
    plot_geometric_means(iterations, bo_geo_means, rs_geo_means, random_geo_means)

if __name__ == "__main__":
    main()
