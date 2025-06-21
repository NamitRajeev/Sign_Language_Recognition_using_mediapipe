import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

#define path
DATA_DIR = r"C:\Users\namit\OneDrive\Desktop\RSET\Other_projects\ISL_3"
CSV_FILE = os.path.join(DATA_DIR, "landmarks.csv")

#load dataset
df = pd.read_csv(CSV_FILE)
print("Loaded CSV with shape:", df.shape)

#separate features and labels
X = df.drop("label", axis=1).values
y = df["label"].values
X = X / 1.0

#Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Save label classes for reference
label_classes_path = os.path.join(DATA_DIR, "label_classes.npy")
np.save(label_classes_path, le.classes_)
print("Saved label classes to:", label_classes_path)

#shuffle data
X, y_encoded = shuffle(X, y_encoded, random_state=42)

#Save preprocessed data
np.save(os.path.join(DATA_DIR, "X.npy"), X)
np.save(os.path.join(DATA_DIR, "y.npy"), y_encoded)

print("Preprocessing complete.")
print("X shape:", X.shape)
print("y shape:", y_encoded.shape)
