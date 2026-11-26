
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/sensor_data.csv")

# Separate features and target
X = data.drop("condition", axis=1).values
y = data["condition"].values

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
np.savez(
    "output/processed_data.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save scaler for future predictions
joblib.dump(scaler, "output/scaler.pkl")

print("DATA PREPROCESSING COMPLETED!")
print("--------------------------------")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nMean of standardized training features:")
print(X_train.mean(axis=0))

print("\nProcessed files saved successfully!")
