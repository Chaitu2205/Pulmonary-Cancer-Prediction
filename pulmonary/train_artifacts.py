# train_artifacts.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# 1. Load dataset
df = pd.read_csv("survey_lung_cancer.csv")  # make sure file name matches
df.columns = [c.strip() for c in df.columns]

# 2. Encode GENDER and target LUNG_CANCER manually
df["GENDER"] = df["GENDER"].map({"FEMALE": 0, "MALE": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# 3. Features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# 4. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Build ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train with early stopping
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[es],
    verbose=1,
)

# 8. Quick evaluation
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {acc:.3f}")

# 9. Save artifacts
model.save("lung_cancer_ann_model.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved files:")
print(" - lung_cancer_ann_model.h5")
print(" - scaler.pkl")
