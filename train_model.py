import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report

#Loading dataset
df = pd.read_csv("C:/Users/DCL/Desktop/Research Paper/diabetes.csv")

#Assigning features and target
features = ["Age", "Gender", "BMI", "SBP", "DBP", "FPG", "Chol", "Tri",
            "HDL", "LDL", "ALT", "BUN", "CCR", "FFPG", "smoking", "drinking", "family_history"]
target = "Diabetes"

X = df[features]
y = df[target]

#Spliting dataset (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

#Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Saving scaler for API
joblib.dump(scaler, "scaler.pkl")

#Computing class weights to handle imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}
print(f"Class weights: {class_weights}")

#Building model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

#Callbacks
early_stop = EarlyStopping(patience=15, restore_best_weights=True)   #patience increased to for longer training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

#Training model(longer training with safeguards)
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,                
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

#Saving model
model.save("diabetes_model.keras")

#Calculating best threshold based on F1 score
y_probs = model.predict(X_test_scaled).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2*(precision * recall)/(precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold for max F1: {best_threshold:.2f}")
y_pred = (y_probs >= best_threshold).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Example prediction
print("Sample prediction (probability):", model.predict(X_test_scaled[0].reshape(1, -1)))
