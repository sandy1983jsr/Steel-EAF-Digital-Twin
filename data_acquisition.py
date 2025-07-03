import random
import time
from datetime import datetime

def get_sensor_data():
    """Simulate data acquisition from blast furnace sensors."""
    return {
        "timestamp": datetime.now().isoformat(),
        "temperature": random.uniform(1200, 1600),      # Celsius
        "pressure": random.uniform(2, 5),               # Bar
        "CO_content": random.uniform(18, 23),           # %
        "feed_rate": random.uniform(50, 70),            # TPH
        "air_flow": random.uniform(800, 1200),          # Nm3/min
        "hot_metal_level": random.uniform(3, 7),        # meters
        "slag_rate": random.uniform(3, 8),              # TPH
    }

def stream_sensor_data(interval=1):
    """Yield sensor data at regular intervals."""
    while True:
        yield get_sensor_data()
        time.sleep(interval)
