import csv
import numpy as np
from tdc import Oracle
from utils.utils import evaluate_objectives

# Create oracles for various objectives
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")

def extract_and_evaluate_smiles(input_csv: str, output_csv: str):
    """
    Extract the chosen SMILES from the input CSV file, evaluate their objectives,
    and write the results to an output CSV file.

    Parameters:
    input_csv (str): Path to the input CSV file containing chosen SMILES.
    output_csv (str): Path to the output CSV file to save the evaluated results.
    """
    chosen_smiles = []

    # Read the chosen SMILES from the input CSV file
    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            smiles = row[3]  # Assuming the chosen SMILES is in the 4th column (index 3)
            chosen_smiles.append(smiles)

    # Evaluate the objectives for each SMILES
    results = evaluate_objectives(chosen_smiles)

    # Write the results to the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SMILES", "f1", "f2", "f3"])
        for i, smiles in enumerate(chosen_smiles):
            writer.writerow([smiles, results[i, 0], results[i, 1], results[i, 2]])

# Example usage
input_csv_path = "dockstring_test.csv"  # Replace with your input CSV file path
output_csv_path = "evaluated_smiles_results.csv"  # Replace with your desired output CSV file path

extract_and_evaluate_smiles(input_csv_path, output_csv_path)
