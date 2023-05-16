import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = keras.models.load_model('cnn_model.h5')

# Load new data for inference (replace this with your new data file)
new_data = np.load('experiment3.npz', allow_pickle=True)

# Access the arrays within the .npz file
new_bandsD = new_data['bandsD']
new_d = new_data['d']
new_epochsLinked = new_data['epochsLinked']

# Combine 'new_bandsD' and 'new_d' arrays for extended features
new_X = np.concatenate((new_bandsD, new_d), axis=1)

# Get the true labels from 'new_epochsLinked'
true_labels = new_epochsLinked[:, 2]

# Reshape new_X to fit the model
new_X = new_X.reshape((new_X.shape[0], new_X.shape[1], 1))

# Make predictions using the trained model
predictions = model.predict(new_X)

# Convert predictions to sleep stage labels (1: wakefulness, 2: NREM sleep, 3: REM sleep)
predicted_labels = np.argmax(predictions, axis=1) + 1

# Compute the confusion matrix
conf_mat = confusion_matrix(true_labels, predicted_labels)

# Define the sleep stage labels
labels = ['Wakefulness', 'NREM Sleep', 'REM Sleep']

# Print the confusion matrix to the console
print("\nConfusion Matrix:")
print(conf_mat)

# Also print a labeled version of the confusion matrix
print("\nConfusion Matrix with Labels:")
for i, row in enumerate(conf_mat):
    print(f"{labels[i]}: {row}")


# Create a heatmap for the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
