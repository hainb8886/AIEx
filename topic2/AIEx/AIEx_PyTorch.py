#GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models, based on PyTorch and MXNet.
#https://ts.gluon.ai/stable/

#pip install gluonts[torch]
#pip install "gluonts[mxnet]"
#pip install orjson

import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.split import split
from gluonts.torch.distributions import StudentTOutput

# Step 1: Generate synthetic time series data
def generate_time_series():
    np.random.seed(42)
    time = np.arange(0, 100, 1)
    series = 10 * np.sin(0.1 * time) + 0.5 * time + np.random.normal(scale=2, size=len(time))
    return series

data = generate_time_series()

# Step 2: Prepare the dataset
train_data = ListDataset(
    [{"start": "2023-01-01", "target": data[:80]}],  # Training set
    freq="1D",
)

test_data = ListDataset(
    [{"start": "2023-01-01", "target": data}],  # Full dataset for testing
    freq="1D",
)

# Step 3: Define the Trainer
#train_data, test_gen = split(data, offset=-36)

# Step 4: Define the DeepAR estimator
estimator = DeepAREstimator(
    prediction_length=10,        # Number of steps to predict into the future
    freq="1D",                   # Frequency of the data
    distr_output=StudentTOutput(), # Distribution for probabilistic forecasts
    num_layers=2,                # Number of LSTM layers
    hidden_size=40,              # Hidden layer size
    dropout_rate=0.1            # Dropout rate
    )

# Step 5: Train the model
predictor = estimator.train(train_data)

# Step 6: Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=100
)

# Step 7: Evaluate the predictions
evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(ts_it, forecast_it)

# Step 8: Display evaluation metrics
print("Aggregate metrics:", agg_metrics)


