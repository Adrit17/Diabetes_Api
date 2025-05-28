# Diabetes Prediction API using Neural Network

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

---

## Overview

This project is a **diabetes prediction system** built with a TensorFlow-based neural network. It takes clinical patient data as input, scales and preprocesses it, then predicts the likelihood of diabetes. The API is built using **FastAPI** to provide real-time predictions.

The model handles common medical data preprocessing tasks including unit conversion (mg/dL to mmol/L), class imbalance, and threshold tuning to maximize F1-score for robust diagnosis.

---

## Features

- Trained deep neural network with dropout and L2 regularization  
- Scales and preprocesses patient clinical data  
- Automatically converts common lab units (mg/dL → mmol/L)  
- Threshold tuning for optimal precision-recall balance  
- Real-time prediction API using FastAPI  
- Easy to extend or retrain with your own dataset  

---

## Files

| Filename       | Description                                    |
|----------------|------------------------------------------------|
| `train_test.py`| Script to train, evaluate, and save the model. |
| `main.py`      | FastAPI app exposing the prediction endpoint.  |
| `scaler.pkl`   | Saved StandardScaler used in preprocessing.    |
| `diabetes_model.keras` | Saved trained neural network model.    |
| `README.md`    | Project overview and instructions.             |

---

## Requirements

- Python 3.8 or higher  
- TensorFlow 2.x  
- scikit-learn  
- FastAPI  
- joblib  
- uvicorn (for running the API)  

Install dependencies:

```bash
pip install tensorflow scikit-learn fastapi joblib uvicorn
