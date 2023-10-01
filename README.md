# Predicting-stocks-
Predicting stocks YouTube 
import pandas as pd

import yfinance as yf

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

# Step 1: Data Collection

# Define the stock symbol and date range

stock_symbol = 'AAPL'

start_date = '2010-01-01'

end_date = '2021-12-31'

# Download historical stock price data

data = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Data Cleaning

# Handle missing values

data.dropna(inplace=True)

# Remove duplicates (if any)

data.drop_duplicates(inplace=True)

# Step 3: Feature Engineering

# Create a feature (X) and target (y)

X = data[['Close']]

y = data['Close'].shift(-1).dropna()

# Step 4: Model Selection
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model

svm_model = SVR(kernel='linear')

svm_model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation

# Make predictions on the test set

y_pred = svm_model.predict(X_test_scaled)

# Calculate RMSE

rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Root Mean Squared Error (RMSE): {rmse}')
