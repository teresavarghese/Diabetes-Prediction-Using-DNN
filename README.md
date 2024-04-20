# Diabetes-Prediction-Using-DNN

This repository contains a Python script for implementing a deep neural network (DNN) model to predict diabetes based on relevant medical features. The script utilizes the Keras library, which is built on top of TensorFlow, to create and train the DNN model. The model is trained on a dataset containing various health indicators such as age, body mass index (BMI), blood pressure, and glucose levels, among others, to predict the likelihood of an individual having diabetes.

## Code Description

The main script, `diabetes_prediction.py`, loads the dataset, preprocesses it, builds a DNN model, compiles it, trains it on the training data, and evaluates its performance on the test data. Key steps include:

1. **Data Handling**: The script loads the `diabetes.csv` dataset using pandas and separates input features (X) and target variable (y).

2. **Preprocessing**: It splits the dataset into training and testing sets, handles missing values, scales features, and encodes categorical variables as needed.

3. **Model Construction**: The DNN model is built using Keras's Sequential API, consisting of multiple Dense layers with ReLU activation, followed by a sigmoid output layer.

4. **Model Compilation**: The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification.

5. **Training**: The compiled model is trained on the training data for a specified number of epochs, adjusting weights and biases to minimize the loss.

6. **Evaluation**: Model performance is evaluated on the test data using metrics like accuracy, precision, recall, and F1-score.

7. **Prediction and Confidence Analysis**: The trained model makes predictions on the test data, and an overall confidence average is calculated, indicating the model's certainty in its predictions.

## Dependencies

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- keras
