import pandas as pd
import yfinance as yf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Step 1: User Input for Stock Symbol
stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA, MSFT): ").upper()
start_date = '2010-01-01'
end_date = '2024-01-01'

# Step 2: Download Historical Stock Data
print(f"Fetching data for {stock_symbol}...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 3: Data Cleaning
if data.empty:
    print("Invalid stock symbol or no data available. Please try again.")
    exit()

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Step 4: Feature Engineering
X = data[['Open', 'High', 'Low', 'Volume']]
y = data[['Close']]  # Keep it as DataFrame, but we'll fix its shape later

# Align features and target
y = y.shift(-1)
X = X[:-1]
y = y.dropna()

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train SVR Model (Fix y_train shape)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train.values.ravel())  # Convert y_train to 1D

# Step 8: Model Evaluation
y_pred = svm_model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually take square root
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Step 9: Predict Next Day's Closing Price
latest_data = scaler.transform([X.iloc[-1]])
next_day_prediction = svm_model.predict(latest_data)[0]
print(f'Predicted Closing Price for {stock_symbol} (Next Trading Day): ${next_day_prediction:.2f}')
