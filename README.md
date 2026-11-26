
# Deep Learning-Based Predictive Maintenance System Using PyTorch

## 1. Project Overview

This project develops a Deep Learning-based Predictive Maintenance System using PyTorch.

The system analyzes machine sensor readings and classifies machine conditions into three categories:

- Normal (0)
- Warning (1)
- Failure (2)

The project demonstrates how neural networks can learn patterns from sensor data to identify potential machine conditions.

## 2. Project Objectives

- Understand PyTorch tensors and neural networks.
- Implement forward propagation and backpropagation.
- Build a custom PyTorch Dataset.
- Use DataLoader for batch processing.
- Implement a Multi-Layer Perceptron (MLP).
- Apply Batch Normalization and Dropout.
- Train the model using Adam optimizer.
- Evaluate model performance.
- Track training loss and accuracy using TensorBoard.

## 3. Technologies Used

- Python 3.13
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorBoard

## 4. Dataset Description

The project uses a synthetic sensor dataset containing 1,000 records.

### Input Features

1. Temperature
2. Vibration
3. Pressure
4. Speed
5. Operating Hours

### Target Classes

| Class | Machine Condition |
|---|---|
| 0 | Normal |
| 1 | Warning |
| 2 | Failure |

**Note:** The dataset is synthetically generated for educational purposes. It does not represent actual industrial sensor measurements.

## 5. Neural Network Architecture

The project uses a Multi-Layer Perceptron (MLP).

| Layer | Architecture |
|---|---|
| Input | 5 features |
| Hidden Layer 1 | Linear (5 → 64) |
| Normalization | BatchNorm1d |
| Activation | ReLU |
| Regularization | Dropout (0.3) |
| Hidden Layer 2 | Linear (64 → 32) |
| Normalization | BatchNorm1d |
| Activation | ReLU |
| Regularization | Dropout (0.2) |
| Output | Linear (32 → 3) |

## 6. Model Training

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Epochs: 30
- Batch Size: 32
- Train/Test Split: 80/20
- Feature Scaling: StandardScaler

The training process uses:
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient optimization

## 7. Model Evaluation

The model achieved the following result on the held-out test dataset:

**Test Accuracy: 94.00%**

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 0.98 | 0.94 | 0.96 |
| Warning | 0.95 | 0.88 | 0.91 |
| Failure | 0.91 | 1.00 | 0.95 |

The confusion matrix and TensorBoard training graphs are included in the output folder.

## 8. Experiment Tracking

TensorBoard is used to visualize:

- Training Loss
- Training Accuracy

TensorBoard records are stored in the `runs/` directory.

## 9. Project Structure

```text
Module_03_Deep_Learning/
│
├── data/
│   ├── generate_dataset.py
│   ├── inspect_dataset.py
│   └── sensor_data.csv
│
├── notebooks/
│
├── output/
│   ├── confusion_matrix.png
│   ├── predictive_maintenance_model.pth
│   ├── processed_data.npz
│   ├── scaler.pkl
│   └── tensorboard_training_graphs.png
│
├── runs/
│   └── TensorBoard event files
│
├── src/
│   ├── backpropagation.py
│   ├── dataset_loader.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── forward_propagation.py
│   ├── neural_network.py
│   ├── optimizer_adam.py
│   ├── optimizer_sgd.py
│   ├── plot_confusion_matrix.py
│   ├── tensor_basics.py
│   ├── tensor_operations.py
│   └── train_model.py
│
├── README.md
└── requirements.txt
```

## 10. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset

```bash
python data/generate_dataset.py
```

### Step 3: Preprocess Dataset

```bash
python src/data_preprocessing.py
```

### Step 4: Train Model

```bash
python src/train_model.py
```

### Step 5: Evaluate Model

```bash
python src/evaluate_model.py
```

### Step 6: View TensorBoard

```bash
python -m tensorboard.main --logdir=runs
```

Open http://localhost:6006/ in your browser.

## 11. Limitations and Future Improvements

- The current dataset is synthetic.
- Real industrial sensor data is required for practical deployment.
- Future work can explore LSTM and CNN architectures.
- The system can be extended with real-time sensor monitoring.
- Further testing is needed before using predictions for maintenance decisions.

## 12. Conclusion

This project demonstrates the implementation of a Deep Learning-based predictive maintenance classification system using PyTorch.

It covers tensor operations, neural network construction, backpropagation, optimization, model evaluation, and experiment tracking using TensorBoard.
