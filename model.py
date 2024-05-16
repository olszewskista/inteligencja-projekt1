import matplotlib.pyplot as plt
import numpy as np
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Flatten
from keras.api.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Prepare the data

keypoints = np.load("keypoints.npy")
labels = np.load("labels.npy")

# print(keypoints.shape, labels.shape)
# print(keypoints, labels)

train_points, test_points, train_labels, test_labels = train_test_split(
    keypoints, labels, train_size=0.8, random_state=1
)

# Convert output labels to categorical one-hot encoding

og_test_labels = np.copy(test_labels)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Define the model
model = Sequential()
model.add(Flatten(input_shape=(6,)))  # Flatten input if necessary
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(5, activation="softmax"))  # 5 classes: up, down, left, right, front

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_checkpoint = ModelCheckpoint(
    filepath='test.model.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

early_stoping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
)

# Train the model
history = model.fit(
    train_points,
    train_labels,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[model_checkpoint, early_stoping],
)

# Evaluate the model

test_loss, test_accuracy = model.evaluate(test_points, test_labels, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.tight_layout()
plt.show()

# Predict test classes
result = model.predict(test_points)

result = np.argmax(result, axis=1)


# Confusion matrix
conf_matrix = confusion_matrix(og_test_labels, result)

ConfusionMatrixDisplay(conf_matrix).plot()

plt.show()