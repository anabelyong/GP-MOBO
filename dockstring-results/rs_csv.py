import re
import csv

# File paths
input_file = 'rs_run_3.txt'
output_file = 'rs_run_3_results.csv'

# Regular expression patterns to extract data
iteration_pattern = re.compile(r'Start Random Sampling iteration (\d+)\.')
smiles_pattern = re.compile(r'Chosen SMILES: (.+?) with value = ([\d.]+)')

# Lists to store extracted data
iterations = []
chosen_smiles = []
smiles_values = []

# Open the input file and read line by line
with open(input_file, 'r') as file:
    for line in file:
        # Match iteration number
        iteration_match = iteration_pattern.search(line)
        if iteration_match:
            iteration = int(iteration_match.group(1))
        
        # Match chosen SMILES and its value
        smiles_match = smiles_pattern.search(line)
        if smiles_match:
            smiles = smiles_match.group(1)
            value = float(smiles_match.group(2))
            
            # Append data to lists
            iterations.append(iteration)
            chosen_smiles.append(smiles)
            smiles_values.append(value)

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Iteration', 'Chosen SMILES', 'Value of Chosen SMILES'])
    for i in range(len(iterations)):
        csvwriter.writerow([iterations[i], chosen_smiles[i], smiles_values[i]])

print(f"Data extracted and saved to {output_file}")
