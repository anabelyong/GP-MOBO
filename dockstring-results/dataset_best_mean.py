import pandas as pd

# Load the CSV file
file_path = 'dataset_best.csv'
df = pd.read_csv(file_path)

# Calculate the mean of the "Chosen SMILES Value"
mean_value = df['Value of Best SMILES'].mean()
print(mean_value)
