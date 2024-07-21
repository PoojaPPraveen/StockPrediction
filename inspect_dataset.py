import pandas as pd

# Load the dataset
stock_data = pd.read_csv('nyse.csv')

# Print the first few rows and columns
print("Columns in the dataset:")
print(stock_data.columns)

print("\nFirst few rows of the dataset:")
print(stock_data.head())
