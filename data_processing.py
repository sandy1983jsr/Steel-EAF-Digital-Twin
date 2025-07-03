import pandas as pd

def clean_data(raw_data: dict) -> dict:
    """Clean and validate raw sensor data."""
    # Remove impossible values, fill missing, etc.
    cleaned = {}
    for k, v in raw_data.items():
        if v is None:
            cleaned[k] = 0.0
        elif k in ['temperature', 'pressure', 'CO_content', 'feed_rate', 'air_flow', 'hot_metal_level', 'slag_rate']:
            cleaned[k] = max(0, v)
        else:
            cleaned[k] = v
    return cleaned

def to_dataframe(data_list):
    """Convert a list of dicts to a pandas DataFrame."""
    return pd.DataFrame(data_list)
