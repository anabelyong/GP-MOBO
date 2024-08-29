import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
ei_csv_file = "evaluated_smiles_results.csv"  # Replace with your EI CSV file path
ehvi_csv_file = "/Users/anabelyong/kernel-mobo/results/bo_results_1.csv"  # Replace with your EHVI CSV file path
#random_csv_file = "/Users/anabelyong/kernel-mobo/results/rs_results_1.csv"  # Replace with your Random Sampling CSV file path
pareto_csv_file = "pareto_points.csv"  # The CSV file we just created

ei_data = pd.read_csv(ei_csv_file)
ehvi_data = pd.read_csv(ehvi_csv_file)
#random_data = pd.read_csv(random_csv_file)
pareto_data = pd.read_csv(pareto_csv_file)

# Extract the necessary columns for plotting
ei_f1, ei_f2, ei_f3 = ei_data["f1"], ei_data["f2"], ei_data["f3"]
ehvi_f1, ehvi_f2, ehvi_f3 = ehvi_data["f1"], ehvi_data["f2"], ehvi_data["f3"]
#random_f1, random_f2, random_f3 = random_data["f1"], random_data["f2"], random_data["f3"]
pareto_f1, pareto_f2, pareto_f3 = pareto_data["f1"], pareto_data["f2"], pareto_data["f3"]

# Function to create pairwise plots
def create_combined_pairwise_plots(ei_f1, ei_f2, ei_f3, ehvi_f1, ehvi_f2, ehvi_f3, pareto_f1, pareto_f2, pareto_f3): #random_f1, random_f2, random_f3,
    plt.figure(figsize=(18, 6))

    # Plot f1 vs f2
    plt.subplot(1, 3, 1)
    plt.scatter(ei_f1, ei_f2, color='orange', label='Dockstring-EI')
    plt.scatter(ehvi_f1, ehvi_f2, color='blue', label='Dockstring-EHVI')
    #plt.scatter(random_f1, random_f2, color='green', label='Random Sampling')
    plt.scatter(pareto_f1, pareto_f2, color='red', label='Pareto Points')
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("f1 vs f2")
    plt.legend()

    # Plot f2 vs f3
    plt.subplot(1, 3, 2)
    plt.scatter(ei_f2, ei_f3, color='orange', label='Dockstring-EI')
    plt.scatter(ehvi_f2, ehvi_f3, color='blue', label='Dockstring-EHVI')
    #plt.scatter(random_f2, random_f3, color='green', label='Random Sampling')
    plt.scatter(pareto_f2, pareto_f3, color='red', label='Pareto Points')
    plt.xlabel("f2")
    plt.ylabel("f3")
    plt.title("f2 vs f3")
    plt.legend()

    # Plot f1 vs f3
    plt.subplot(1, 3, 3)
    plt.scatter(ei_f1, ei_f3, color='orange', label='Dockstring-EI')
    plt.scatter(ehvi_f1, ehvi_f3, color='blue', label='Dockstring-EHVI')
    #plt.scatter(random_f1, random_f3, color='green', label='Random Sampling')
    plt.scatter(pareto_f1, pareto_f3, color='red', label='Pareto Points')
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.title("f1 vs f3")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Create the combined pairwise plots
create_combined_pairwise_plots(ei_f1, ei_f2, ei_f3, ehvi_f1, ehvi_f2, ehvi_f3, pareto_f1, pareto_f2, pareto_f3)
#random_f1, random_f2, random_f3,