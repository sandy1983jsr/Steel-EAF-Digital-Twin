import streamlit as st
import pandas as pd
from data_acquisition import get_sensor_data
from data_processing import clean_data, to_dataframe
from blast_furnace_model import BlastFurnaceSimulator, detect_anomaly

st.set_page_config(page_title="Arjas Steel Blast Furnace Digital Twin", layout="wide")
st.title("Arjas Steel - Blast Furnace Digital Twin Dashboard")

model = BlastFurnaceSimulator()

# File uploader for CSV data
uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample (simulated) data", value=True if not uploaded_file else False)

# Data acquisition logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Try to ensure column names match; if not, show warning
    expected_cols = {"timestamp", "temperature", "pressure", "CO_content", "feed_rate", "air_flow", "hot_metal_level", "slag_rate"}
    uploaded_cols = set(df.columns)
    if not expected_cols.issubset(uploaded_cols):
        st.warning(
            f"Some expected columns are missing from the uploaded CSV.\n"
            f"Expected: {expected_cols}\nFound: {uploaded_cols}"
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    history = df.to_dict("records")
elif use_sample or not uploaded_file:
    # Use or initialize session state for simulated data
    if 'sensor_history' not in st.session_state:
        st.session_state['sensor_history'] = []
    history = st.session_state['sensor_history']
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    if not uploaded_file:
        if st.button("Get Latest Sensor Data"):
            raw = get_sensor_data()
            clean = clean_data(raw)
            st.session_state['sensor_history'].append(clean)
            st.success("New sensor data ingested!")
            history = st.session_state['sensor_history']

    if history:
        df = to_dataframe(history)
        st.dataframe(df.tail(20))

        # Show trends
        st.line_chart(df.set_index('timestamp')[['temperature', 'pressure', 'CO_content']])
        st.line_chart(df.set_index('timestamp')[['feed_rate', 'air_flow', 'hot_metal_level', 'slag_rate']])
    else:
        st.info("Click the button to get sensor data or upload a CSV.")

with col2:
    if history:
        latest = history[-1]
        # Simulate
        pred = model.simulate_step(latest)
        st.metric("Predicted Hot Metal Output (TPH)", f"{pred['predicted_hot_metal']:.2f}")

        # Anomaly detection
        anomaly = detect_anomaly(latest)
        if anomaly:
            st.error(f"Anomalies detected: {anomaly}")
        else:
            st.success("No anomalies detected.")

st.markdown("---")
st.caption("© Arjas Steel Digital Twin Example")
