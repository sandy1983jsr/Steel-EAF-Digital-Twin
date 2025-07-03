import numpy as np

class BlastFurnaceSimulator:
    """
    Simple simulation model for a blast furnace.
    You can expand this with physical/chemical equations.
    """

    def __init__(self):
        pass

    def predict_hot_metal_output(self, feed_rate, air_flow, temperature, pressure):
        # Example empirical formula (for demo purpose)
        efficiency = min(1.0, (temperature - 1200) / 400 * 0.9 + 0.1)
        output = feed_rate * efficiency * (air_flow / 1000) * (pressure / 3)
        return output

    def simulate_step(self, sensor_data: dict):
        output = self.predict_hot_metal_output(
            feed_rate=sensor_data['feed_rate'],
            air_flow=sensor_data['air_flow'],
            temperature=sensor_data['temperature'],
            pressure=sensor_data['pressure'],
        )
        return {
            "predicted_hot_metal": output
        }

# Example anomaly detection (can be replaced with ML model)
def detect_anomaly(sensor_data: dict) -> dict:
    anomalies = {}
    if sensor_data['temperature'] < 1250 or sensor_data['temperature'] > 1550:
        anomalies['temperature'] = 'out_of_range'
    if sensor_data['pressure'] < 2.2 or sensor_data['pressure'] > 4.8:
        anomalies['pressure'] = 'out_of_range'
    # Add more rules as needed
    return anomalies
