''' 1. Synthetic Data Generation:
Generates a time series with sine waves and noise.

2. Dataset Preparation:
Uses ListDataset from GluonTS to define train and test datasets.

3. Model Definition:
Builds an LSTM-based TensorFlow model for forecasting.

4. Training:
The model is trained using a Mean Squared Error (MSE) loss function.

5. Evaluation:
Uses GluonTS utilities to evaluate the model’s predictions against actual data. '''

''' example demonstrates how to use GluonTS to prepare a time series dataset and TensorFlow to build and train a forecasting model.
export TF_ENABLE_ONEDNN_OPTS=0  # Linux/MacOS
set TF_ENABLE_ONEDNN_OPTS=0    # Windows
'''
import tensorflow as tf
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

# Step 1: Generate synthetic data
def generate_time_series():
    np.random.seed(42)
    time = np.arange(0, 100, 1)
    series = 10 * np.sin(0.1 * time) + 0.5 * time + np.random.normal(scale=2, size=len(time))
    return series


data = generate_time_series()
#print("Data :", data)

# Step 2: Prepare dataset for GluonTS
train_data = ListDataset(
    [{"start": "2023-01-01", "target": data[:80]}],
    freq="1D"
)

test_data = ListDataset(
    [{"start": "2023-01-01", "target": data}],
    freq="1D"
)

# Step 3: Define a TensorFlow forecasting model
class TimeSeriesModel(tf.keras.Model):
    def __init__(self, lookback, forecast_horizon):
        super(TimeSeriesModel, self).__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        self.dense = tf.keras.layers.Dense(forecast_horizon)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)

# Step 4: Prepare input data
lookback = 10
forecast_horizon = 10

def prepare_data(data, lookback, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        x.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(x), np.array(y)

train_series = to_pandas(list(train_data)[0])
x_train, y_train = prepare_data(train_series.values, lookback, forecast_horizon)

# Step 5: Train the model
model = TimeSeriesModel(lookback, forecast_horizon)
model.compile(optimizer="adam", loss="mse")

x_train = np.expand_dims(x_train, axis=-1)  # Add feature dimension
model.fit(x_train, y_train, epochs=20, batch_size=16)

# Step 6: Make Predictions:
# Prepare Test Data:
x_test, _ = prepare_data(data, lookback, forecast_horizon)
x_test = np.expand_dims(x_test, axis=-1)

# Generate Predictions:
predictions = model.predict(x_test)

# Step 7: Evaluate the Model:

# Use GluonTS Evaluator:
evaluator = Evaluator()
forecast_it, ts_it = make_evaluation_predictions(test_data,
                                                  lambda batch: predictions[:, :forecast_horizon],
                                                  num_samples=100)
agg_metrics, item_metrics = evaluator(ts_it, forecast_it)

print("Aggregate metrics:", agg_metrics)
