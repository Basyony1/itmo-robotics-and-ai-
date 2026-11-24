# House Price Prediction Using Neural Networks

## Overview

This project implements a machine learning solution for predicting house prices using the Ames Housing dataset.

The main objective is to explore and compare different regression approaches, starting with Linear Regression and then developing a Neural Network using TensorFlow and Keras.

The project also investigates how different hyperparameters affect model performance.

## Objectives

- Perform data preprocessing and exploratory data analysis.
- Handle missing values and categorical features.
- Apply feature scaling using `StandardScaler`.
- Build a baseline Linear Regression model.
- Develop a Neural Network for regression.
- Experiment with different hyperparameters.
- Evaluate and compare model performance.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Dataset

The project uses the Ames Housing dataset, which contains information about residential properties and their sale prices.

The target variable is:

- `SalePrice` — the sale price of a house.

The input features include property characteristics such as living area, number of rooms, construction year, neighborhood, and other housing attributes.

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the training and test datasets.
2. Handling missing values.
3. Removing columns with a large number of missing values.
4. Encoding categorical variables.
5. Separating features (`X`) from the target (`y`).
6. Splitting the data into training and validation sets.
7. Applying `StandardScaler` to the input features and target variable.

The dataset is split using:

- Training set: 80%
- Validation set: 20%
- Random state: 40

## Models

### 1. Linear Regression

A Linear Regression model is implemented as a baseline to establish a reference performance.

### 2. Neural Network Regression

A feedforward Neural Network is developed using TensorFlow/Keras.

The initial architecture consists of:

- Input Layer: 74 features
- Hidden Layer 1: 64 neurons, ReLU activation
- Hidden Layer 2: 32 neurons, ReLU activation
- Output Layer: 1 neuron

The model uses the Adam optimizer and Mean Squared Error (MSE) as its initial loss function.

## Hyperparameter Experiments

Several experiments are performed to investigate the impact of different hyperparameters:

| Hyperparameter | Tested Values |
|---|---|
| Number of neurons | 16, 64, 256 |
| Hidden layer size | 0, 32, 128 |
| Epochs | 10, 50, 100, 200 |
| Batch size | 16, 32, 128 |
| Optimizer | Adam, SGD, RMSprop |
| Loss function | MSE, MAE, Huber |

These experiments help evaluate how model architecture and training configurations influence prediction accuracy.

## Final Model Configuration

The selected final Neural Network configuration is:

| Parameter | Value |
|---|---|
| First hidden layer | 256 neurons |
| Second hidden layer | 32 neurons |
| Activation function | ReLU |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | Adam |
| Loss function | MAE |

## Evaluation Metrics

The models are evaluated using the following regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Represents the square root of MSE.
- **R² Score:** Measures how well the model explains the variance in house prices.

## Results

The final Neural Network achieved the following reported validation performance:

| Metric | Result |
|---|---:|
| MAE | $14,820 |
| R² Score | 0.894 |

The experiments demonstrate how neural network architecture and training hyperparameters can influence regression performance.

## Project Structure

```text
.
├── README.md
├── train.csv
├── test.csv
└── house_price_prediction.ipynb
```

*The file names and structure above should be adjusted to match the actual repository.*

## Author

**Abdelsalam Basyony**

ITMO University

Faculty of Control Systems and Robotics

## Academic Context

This project was completed as part of a practical assignment on Neural Networks and Regression.
