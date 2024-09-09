import re
import csv

# Define the input and output file paths
input_file = 'tanimoto_gpbo_ei_run_1.txt'
output_file = 'tanimoto_gpbo_ei_run_1.csv'

# Regular expressions to extract the necessary information
iteration_re = re.compile(r'Start BO iteration (\d+)\.')
acq_value_re = re.compile(r'Eval batch acq values: \[(.*?)\]')
smiles_re = re.compile(r'Eval batch SMILES: \[(.*?)\]')
value_re = re.compile(r'Value of chosen SMILES: ([\d\.]+)')

# Open the input file and create a list to store the parsed data
results = []

with open(input_file, 'r') as f:
    lines = f.readlines()
    
    iteration = None
    acquisition_value = None
    smiles = None
    value = None
    
    for line in lines:
        # Extract iteration number
        iteration_match = iteration_re.search(line)
        if iteration_match:
            iteration = int(iteration_match.group(1))
        
        # Extract acquisition function value
        acq_value_match = acq_value_re.search(line)
        if acq_value_match:
            acquisition_value = float(acq_value_match.group(1))
        
        # Extract chosen SMILES
        smiles_match = smiles_re.search(line)
        if smiles_match:
            smiles = smiles_match.group(1).replace("'", "")
        
        # Extract the value of chosen SMILES
        value_match = value_re.search(line)
        if value_match:
            value = float(value_match.group(1))
        
        # If all values for an iteration are extracted, save them to the results list
        if iteration is not None and acquisition_value is not None and smiles is not None and value is not None:
            results.append([iteration, acquisition_value, smiles, value])
            
            # Reset the variables for the next iteration
            iteration = None
            acquisition_value = None
            smiles = None
            value = None

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow(['Iteration', 'Acquisition Function Value', 'Chosen SMILES', 'Value of Chosen SMILES'])
    # Write the rows
    csvwriter.writerows(results)

print(f"Data has been successfully extracted and saved to {output_file}.")
