import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os

# Load preprocessed data
BASE_PATH = "C:/Users/namit/OneDrive/Desktop/RSET/Other_projects/ISL_3"
X = np.load(os.path.join(BASE_PATH, "X.npy"))
y = np.load(os.path.join(BASE_PATH, "y.npy"))
label_classes = np.load(os.path.join(BASE_PATH, "label_classes.npy"), allow_pickle=True)

# Convert labels to one-hot encoding
y_cat = to_categorical(y, num_classes=len(label_classes))

# Split into training and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_classes), activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=8)

# Save model
model.save(os.path.join(BASE_PATH, "isl_landmark_model.h5"))
print("✅ Model trained and saved as 'isl_landmark_model.h5'")
