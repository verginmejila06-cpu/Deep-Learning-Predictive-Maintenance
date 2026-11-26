
import pandas as pd
import numpy as np

# Reproducible random data
np.random.seed(42)

# Generate sensor readings
samples = 1000

temperature = np.random.uniform(40, 120, samples)
vibration = np.random.uniform(0.1, 10, samples)
pressure = np.random.uniform(50, 150, samples)
speed = np.random.uniform(500, 3000, samples)
operating_hours = np.random.uniform(100, 10000, samples)

# Assign machine condition
conditions = []

for i in range(samples):
    if temperature[i] > 100 or vibration[i] > 8:
        conditions.append(2)  # Failure

    elif temperature[i] > 80 or vibration[i] > 5:
        conditions.append(1)  # Warning

    else:
        conditions.append(0)  # Normal

# Create DataFrame
data = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "speed": speed,
    "operating_hours": operating_hours,
    "condition": conditions
})

# Save dataset
data.to_csv("data/sensor_data.csv", index=False)

print("Dataset generated successfully!")
print("\nFirst 5 records:")
print(data.head())

print("\nCondition distribution:")
print(data["condition"].value_counts().sort_index())

print("\nDataset shape:", data.shape)
