import numpy as np
import pandas as pd
import tensorflow
import joblib
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/DCL/Desktop/Research Paper/diabetes.csv")


features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]

target = "Result"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler,pkl")

model = keras.Sequential([keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],),
                          kernel_regularizer= l2(0.01)),
                          keras.layers.Dense(8, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

model.save("diabetes_model.h5")

print("Model training is completed & saved as diabetes_model.h5")
