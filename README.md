# Diabetes_Api
Main.py
1. Key Components:

Imports:

- FastAPI: Framework for building the API.
- numpy: Handles numerical operations.
- tensorflow: Loads the pre-trained deep learning model.
- joblib: Loads the data scaler for preprocessing.
- pydantic.BaseModel: Defines the expected input data format. 

  ![Screenshot 2025-03-12 202542](https://github.com/user-attachments/assets/7eddfd7d-9464-4ec1-aaad-27781d218145)


2. Model and Scaler Loading:

- The pre-trained deep learning model (diabetes_model.h5) is loaded using TensorFlow.
- The scaler.pkl file (saved with joblib) is loaded for normalizing input data.
  
  ![Screenshot 2025-03-12 203637](https://github.com/user-attachments/assets/86db9f09-dbaa-4dec-9f0d-617bd5378495)

3. API Setup:

- Root Endpoint (/): Returns a simple message to indicate the API is running.

  ![Screenshot 2025-03-12 204543](https://github.com/user-attachments/assets/9307cf4c-3523-4190-bc97-25807314bf87)

- Prediction Endpoint (/predict):
  i/ Accepts JSON input containing patient health metrics.
  ii/ Preprocesses the input using the loaded scaler.
  iii/ Feeds the processed input into the deep learning model for prediction.
  iv/ Returns a prediction result ("Diabetic" or "Not Diabetic") along with confidence score.

  ![Screenshot 2025-03-12 204718](https://github.com/user-attachments/assets/4944010b-6d80-418f-aa4d-834cea3e744e)



Train_Model.py

1. Key imports: 


![Screenshot 2025-03-12 205109](https://github.com/user-attachments/assets/bf8ec03f-4b8d-4404-932d-bbae1b74b2b9)


2. Loading and Preparing Data

- Reads diabetes.csv dataset.
- Selects key health features (Glucose, BMI, age).
- Splits data into training (80%) and testing (20%) sets.
  
  ![Screenshot 2025-03-12 205521](https://github.com/user-attachments/assets/3e61d83b-def4-46bf-8bdd-2cf6bb4ee8ea)


3. Data Preprocessing

 - Uses StandardScaler to normalize input features.
 - Saves the trained scaler as scaler.pkl for future use.
  
   ![Screenshot 2025-03-12 205934](https://github.com/user-attachments/assets/bd0691b9-c94e-4574-8aad-8a16c2bf7631)


4. Building the Neural Network:
   
 A sequential model with:
  - 16 neurons (ReLU activation, L2 regularization)
  - 8 neurons (ReLU activation)
  - 1 neuron (Sigmoid activation for binary classification)
  - Compiled with Adam optimizer and binary cross-entropy loss.

    ![Screenshot 2025-03-12 210512](https://github.com/user-attachments/assets/c9341339-1034-4c9c-a83c-2a8e9a5a7e6a)


5. Training the Model:
   - Runs for 20 epochs with batch size 8.
   - Uses validation data to track performance.
  
    ![Screenshot 2025-03-12 210900](https://github.com/user-attachments/assets/d8480f6f-cdb9-44ab-9aaa-8bed4e1b6a75)



6. Saving the Model:
   - Saves the trained model as diabetes_model.h5.
   - Prints confirmation message when training is complete.
  

     ![Screenshot 2025-03-12 211212](https://github.com/user-attachments/assets/5ac8373d-9494-4246-a1fd-18ef3680521e)



 
