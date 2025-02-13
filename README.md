📈 Stock Price Prediction Using Machine Learning
This Python script predicts the next day's closing price of any stock based on historical market data. It uses Machine Learning (Support Vector Regression - SVR) to analyze stock trends and make predictions.

🔍 How It Works
1️⃣ User Input
The user enters a stock symbol (e.g., AAPL for Apple, TSLA for Tesla).
The program fetches historical stock price data from Yahoo Finance.
2️⃣ Data Collection & Cleaning
The script downloads stock data from 2010 to 2024.
Removes missing values and duplicates to ensure clean data.
3️⃣ Feature Engineering
Uses Open, High, Low, and Volume as input features (X).
Uses Closing Price as the target (y).
The target variable (y) is shifted by one day to predict the next day’s closing price.
4️⃣ Data Preprocessing
Splits the data into training (80%) and testing (20%) sets.
Uses StandardScaler to normalize the feature values.
5️⃣ Model Training (Support Vector Regression - SVR)
The SVR model is trained on the processed data.
It learns patterns and relationships between stock features.
6️⃣ Model Evaluation
The model predicts stock prices on the test set.
Root Mean Squared Error (RMSE) is calculated to measure accuracy.
7️⃣ Next-Day Prediction
The model predicts the closing price for the next trading day based on the latest available data.
