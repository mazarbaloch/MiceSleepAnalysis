import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import wandb

# Configure WandB with your API key
wandb.login(key='b15079c6017deb461bf6e17500c26d80168688f3')
wandb.init(project="SleepAnalysis", entity="research-uef")

#model = keras.models.load_model('trained_model.h5')
data = np.load('experiment1.npz', allow_pickle=True)

# Access the arrays within the .npz file
bandsD = data['bandsD']
d = data['d']
epochsLinked = data['epochsLinked']
epochTime = data['epochTime']

# Combine 'bandsD' and 'd' arrays for extended features
# If 'bandsD' is not used in the paper, simply remove it
X = np.concatenate((bandsD, d), axis=1)

# The target is the sleep stage ID from 'epochsLinked'
y = to_categorical(epochsLinked[:, 2] - 1, num_classes=3)

# The architecture of your model is commented out because you are loading a pre-trained model.
# If you want to use this architecture, uncomment the lines below and comment the 'load_model' line above.
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape X to fit the model
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train the model
for epoch in range(50):
    # Fit the model for one epoch
    history = model.fit(X, y, batch_size=32, epochs=1, validation_split=0.2)

    # Log metrics to WandB
    wandb.log({"epoch": epoch, "train_loss": history.history['loss'][0], "train_acc": history.history['accuracy'][0],
               "val_loss": history.history['val_loss'][0], "val_acc": history.history['val_accuracy'][0]})

# Save the trained model
model.save('cnn_model.h5')
