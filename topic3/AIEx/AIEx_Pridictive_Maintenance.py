#https://scikit-learn.org/stable/
# NumPy, SciPy và Matplotlib, Scikit-learn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Step 1: Simulate Sensor Data
# Generates synthetic sensor data with periodic anomalies and failure labels.
# Failures are rare events (2% probability).
def generate_synthetic_data(num_samples=1000, num_sensors=5):
    np.random.seed(42)
    time = np.arange(num_samples)

    # Generate sensor readings with anomalies
    data = {
        f"sensor_{i + 1}": np.sin(0.1 * time + np.random.rand()) +
                           np.random.normal(scale=0.1, size=num_samples) +
                           (np.random.rand(num_samples) > 0.95) * np.random.uniform(-5, 5, size=num_samples)
        for i in range(num_sensors)
    }

    # Generate failure labels (1 = failure, 0 = normal)
    data['failure'] = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])

    return pd.DataFrame(data)


data = generate_synthetic_data()


# Step 2: Feature Engineering
# Adds rolling statistics (mean, std, min, max) for each sensor to capture temporal patterns.
def create_rolling_features(data, window_size=5):
    feature_data = data.copy()
    for col in data.columns[:-1]:  # Exclude the 'failure' column
        feature_data[f"{col}_mean"] = data[col].rolling(window=window_size, min_periods=1).mean()
        feature_data[f"{col}_std"] = data[col].rolling(window=window_size, min_periods=1).std()
        feature_data[f"{col}_min"] = data[col].rolling(window=window_size, min_periods=1).min()
        feature_data[f"{col}_max"] = data[col].rolling(window=window_size, min_periods=1).max()
    return feature_data.dropna()


data_with_features = create_rolling_features(data)

# Step 3: Split Data into Train and Test Sets
X = data_with_features.drop(columns=["failure"])
y = data_with_features["failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train a Random Forest Classifier
# A Random Forest Classifier is trained to classify whether a failure will occur based on sensor readings.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
#  Model accuracy is evaluated using a confusion matrix and classification report.
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Anomaly Detection (Optional)
# Use the trained model's prediction probabilities to flag potential anomalies
# Flags potential failures using prediction probabilities from the model.
probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.5  # You can tune this threshold
anomalies = X_test[probabilities > threshold]

print("\nAnomalies Detected:")
print(anomalies.head())

# Step 7: Visualization
# Plots sensor readings and highlights actual failure points.
plt.figure(figsize=(10, 6))
plt.plot(data["sensor_1"], label="Sensor 1", alpha=0.7)
plt.scatter(data.index[data["failure"] == 1], data["sensor_1"][data["failure"] == 1],
            color="red", label="Failures", zorder=5)
plt.title("Sensor 1 Readings with Failures")
plt.xlabel("Time")
plt.ylabel("Sensor Reading")
plt.legend()
plt.show()
