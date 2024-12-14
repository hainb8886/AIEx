import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Load or Generate Time Series Data:
def generate_time_series(length=100):
    np.random.seed(100)
    time = np.arange(length)
    series = np.sin(0.1 * time) + np.random.randn(length) * 0.1
    return series

time_series = generate_time_series()
print("time_series: ", time_series)

#Create Training and Validation Sets:
split_time = 80
time_train = time_series[:split_time]
time_valid = time_series[split_time:]

#Create a Sequential Model:
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1]),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

#Compile the Model:
model.compile(loss='mse', optimizer='adam')

#Train the Model:
history = model.fit(time_train, time_train, epochs=100, validation_data=(time_valid, time_valid))

#Make Predictions:
forecast = model.predict(time_valid)

#Visualize Predictions:
plt.plot(time_valid, forecast, label='Prediction')
plt.plot(time_valid, time_valid, label='Actual')
plt.legend()
plt.show()