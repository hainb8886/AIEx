import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Step 1: Data Collection
# Load data
data = pd.read_csv("sample_traansaction.csv")

#Step 2: Preprocessing & Feature Engineering
# Feature Engineering
data['transaction_hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['amount_log'] = data['amount'].apply(lambda x: np.log1p(x))  # Log-transform skewed amounts

# Encode categorical features
encoder = LabelEncoder()
data['location_encoded'] = encoder.fit_transform(data['location'])
data['transaction_type_encoded'] = encoder.fit_transform(data['transaction_type'])

# Normalize features
scaler = StandardScaler()
data[['amount_log', 'transaction_hour']] = scaler.fit_transform(data[['amount_log', 'transaction_hour']])

#Step 3: Model Development
#Supervised Learning (Random Forest Example):
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define features and target
X = data[['amount_log', 'transaction_hour', 'location_encoded', 'transaction_type_encoded']]
y = data['fraud_label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))