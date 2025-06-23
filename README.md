## Diabetes Prediction API using Deep Neural Network

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
| `README.md`    | Project overview and instructions.             |
| `main.py`      | FastAPI app exposing the prediction endpoint.  |
| `train_test.py`| Script to train, evaluate, and save the model. |

---

## Requirements

- Python 3.8 or higher  
- TensorFlow 2.x  
- scikit-learn  
- FastAPI  
- joblib  
- uvicorn (for running the API)  

Install dependencies:

## Training Process (train_model)

Here the model training progress including epochs, loss & accuracy metrics are shown
This GIF shows how the model learns and converges.

![train_test py](https://github.com/user-attachments/assets/3723f4f3-ec55-4ccf-ac95-a88d6a7e2592)

---

## FastAPI Backend (main.py)

This GIF demonstrates the API startup and sample prediction request handling in real-time.

![main py](https://github.com/user-attachments/assets/1b1a00d9-5f10-429b-9d0b-adc23e483997)


---

## Swagger UI Demo

An interactive demo of the FastAPI Swagger UI where the `/predict` endpoint can be tested live.

![swagger_ui](https://github.com/user-attachments/assets/e4879e68-7bbc-409d-bd5f-a1219ad3c57b)


## Required Downloades
```bash
pip install tensorflow scikit-learn fastapi joblib uvicorn

