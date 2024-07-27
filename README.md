# Microsoft-Stock-Price-Prediction

## Overview

This project aims to predict Microsoft stock prices using historical data and machine learning techniques. The model uses Long Short-Term Memory (LSTM) networks, which are well-suited for time series prediction tasks.

## Table of Contents

-Introduction
- Installation
- Data Description
- Exploratory Data Analysis
- Model Building
- Results
- Usage
  
## Introduction

In this project, we use TensorFlow and Keras to build a predictive model for Microsoft stock prices based on historical stock data. The model is designed to forecast future stock prices based on past trends and patterns.

## Installation

- Ensure you have the necessary libraries installed. You can install them using pip:

pip install numpy pandas matplotlib seaborn tensorflow scikit-learn

## Data Description

The dataset used in this project is the Microsoft stock price data and contains the following columns:

- index: Index of the entry.
- date: Date of the record.
- open: Opening price of the stock.
- high: Highest price of the stock on that day.
- low: Lowest price of the stock on that day.
- close: Closing price of the stock.
- volume: Volume of stocks traded.
- Name: Stock name (MSFT in this case).
  
## Sample Data

index	date	open	high	low	close	volume	Name
0	2013-02-08	27.35	27.71	27.31	27.55	33318306	MSFT
1	2013-02-11	27.65	27.92	27.50	27.86	32247549	MSFT

## Exploratory Data Analysis

We performed an exploratory data analysis (EDA) to understand the dataset better. This includes:

- Plotting the opening and closing prices over time.
- Visualizing the trading volume.
- Analyzing correlations between numeric features.

## Model Building

### Data Preparation

- Scaling Data: The data is standardized using StandardScaler.
- Creating Training Data: Sequences of 60 days of stock prices are used to predict the next day's price.

### Model Architecture
The LSTM model consists of:

- LSTM Layers: Two LSTM layers with 64 units each.
- Dense Layers: A Dense layer with 128 units followed by a Dropout layer.
- Output Layer: A Dense layer with a single unit to predict the closing price.

### Model Compilation
The model is compiled using:

- Optimizer: Adam
- Loss Function: Mean Absolute Error (MAE)
- Metric: Root Mean Squared Error (RMSE)

### Training
The model is trained for 20 epochs with the following performance metrics:

- Epoch 1: Loss = 0.3458, RMSE = 0.5245
- Epoch 20: Loss = 0.0794, RMSE = 0.1085

## Results

### Model Evaluation

The trained model is evaluated on test data to assess its performance. Visualization of the training and test data can be found in the results section.

### Example Visualization

plt.figure(figsize=(10, 8))
plt.plot(train['close'], c="b")
plt.title('Microsoft Stock Close Price')
plt.ylabel("Close")

## Usage
To use the model, follow these steps:

- Prepare Data: Ensure you have the dataset in the correct format.
- Run the Script: Execute the Python script to train the model and make predictions.
