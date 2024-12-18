'''
FileName: CryptocurrencyTransaction.csv
Features Explained:
wallet_address: Unique address of the sender/receiver.
timestamp: Date and time of the transaction.
amount: Amount of cryptocurrency transferred.
token: Type of token (ETH, BTC, USDT, etc.).
transaction_type: E.g., transfer, smart_contract interaction, etc.
gas_fee: Transaction fee.
balance: Wallet balance after the transaction.
is_fraud: Label (1 for fraud, 0 for legitimate).'''

'''Feature Engineering
We’ll engineer features to identify anomalies:
Transaction Amount: Log transformation for large outliers.
Transaction Frequency: Count transactions from the same wallet within 24 hours.
Wallet Balance: Low balance with large transactions is suspicious.
Gas Fee Proportion: High gas fees may indicate irregular transactions.'''


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data
# Example cryptocurrency transaction dataset
data = pd.DataFrame({
    "wallet_address": ["0xA1B2C3", "0xD4E5F6", "0xG7H8I9", "0xJ1K2L3", "0xA1B2C3"],
    "timestamp": ["2024-06-14 10:00:00", "2024-06-14 10:05:00",
                  "2024-06-14 11:00:00", "2024-06-14 12:00:00",
                  "2024-06-14 13:00:00"],
    "amount": [1.2, 100.5, 0.001, 50, 0.002],
    "token": ["ETH", "BTC", "USDT", "ETH", "USDT"],
    "transaction_type": ["transfer", "transfer", "smart_contract", "transfer", "transfer"],
    "gas_fee": [0.001, 0.0005, 0.002, 0.005, 0.001],
    "balance": [10, 500, 0.01, 100, 0.005],
    "is_fraud": [0, 1, 0, 1, 0]
})

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 2. Feature Engineering
# Log-transform transaction amount to handle skewness
data['log_amount'] = np.log1p(data['amount'])

# Calculate transactions per wallet in the last 24 hours
data['transaction_count'] = data.groupby('wallet_address')['wallet_address'].transform('count')

# Encode categorical features (token, transaction_type)
label_encoder = LabelEncoder()
data['token_encoded'] = label_encoder.fit_transform(data['token'])
data['transaction_type_encoded'] = label_encoder.fit_transform(data['transaction_type'])

# Final feature set
features = ['log_amount', 'transaction_count', 'gas_fee', 'balance', 'token_encoded', 'transaction_type_encoded']
X = data[features]
y = data['is_fraud']

# Split Data and Train Supervised Model
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Anomaly Detection for unsupervised fraud detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Anomaly Detection with Isolation Forest
# Predict anomalies (-1 = anomaly, 1 = normal)
anomaly_predictions = iso_forest.predict(X_test)
print(f"Anomalies detected: {sum(anomaly_predictions == -1)}")
