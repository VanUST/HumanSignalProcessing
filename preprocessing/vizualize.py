import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Define the Kalman filter class in Python
class KalmanFilter:
    def __init__(self, process_noise, measurement_noise, estimated_error, initial_value):
        self.q = process_noise
        self.r = measurement_noise
        self.p = estimated_error
        self.x = initial_value
        self.k = 0

    def process(self, measurement):
        # Prediction update
        self.p = self.p + self.q

        # Measurement update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p

        return self.x


# Load the uploaded CSV file
file_path = 'D:\HandyRobotics\example_input.csv'

# Read the CSV file into a pandas dataframe
data = pd.read_csv(file_path, header=None)


# Extract the original values
original_values = data[0].dropna().values.astype(float)
original_values[0] = 100

# Initialize the Kalman filter with arbitrary parameters
kf = KalmanFilter(process_noise=1E-5, measurement_noise=10, estimated_error=10, initial_value=original_values[0])

# Apply the Kalman filter to the data
filtered_values = np.array([kf.process(value) for value in original_values])

# Plotting the original and filtered data
plt.figure(figsize=(10, 6))
plt.plot(original_values, label='Original Values', linestyle='--', color='blue')
plt.plot(filtered_values, label='Kalman Filtered Values', color='red')
plt.title('Original vs Kalman Filtered Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
