
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

# Actual and predicted labels from your evaluation
actual = np.array(
    [0] * 48 + [1] * 72 + [2] * 80
)

predicted = np.array(
    [0] * 45 + [1] * 3 +
    [0] * 1 + [1] * 63 + [2] * 8 +
    [2] * 80
)

# Create confusion matrix
cm = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1, 2]
)

# Plot graph
plt.figure(figsize=(7, 5))

plt.imshow(cm, cmap="Blues")

plt.title("Predictive Maintenance - Confusion Matrix")
plt.xlabel("Predicted Condition")
plt.ylabel("Actual Condition")

classes = ["Normal", "Warning", "Failure"]

plt.xticks(range(3), classes)
plt.yticks(range(3), classes)

for i in range(3):
    for j in range(3):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "output/confusion_matrix.png",
    dpi=300
)

plt.show()

print("Confusion matrix graph saved successfully!")
