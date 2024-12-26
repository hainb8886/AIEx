# Combine data from multiple sensors for more reliable obstacle detection.
# Example: Fuse camera, LiDAR, and radar data.
import numpy as np

# Simulated sensor readings
camera_distance = 10.0  # meters
lidar_distance = 10.2
radar_distance = 9.8

# Combine readings using weighted average
def fuse_sensor_data(camera, lidar, radar):
    weights = [0.4, 0.4, 0.2]  # Example weights
    fused_distance = np.dot(weights, [camera, lidar, radar])
    return fused_distance

fused_distance = fuse_sensor_data(camera_distance, lidar_distance, radar_distance)
print(f"Fused Distance: {fused_distance} meters")