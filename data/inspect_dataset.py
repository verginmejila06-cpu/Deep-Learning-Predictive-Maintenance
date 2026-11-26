
import pandas as pd

# Load dataset
data = pd.read_csv("data/sensor_data.csv")

print("DATASET INFORMATION")
print("-------------------")

# Dataset dimensions
print("Shape:", data.shape)

# Column names and data types
print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate records:", data.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Condition distribution
print("\nMachine condition counts:")
print(data["condition"].value_counts().sort_index())
