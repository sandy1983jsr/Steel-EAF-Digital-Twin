import streamlit as st
import pandas as pd
from data_acquisition import get_sensor_data
from data_processing import clean_data, to_dataframe
from blast_furnace_model import BlastFurnaceSimulator, detect_anomaly
from what_if_analysis import what_if_interface

st.set_page_config(page_title="Arjas Steel Blast Furnace Digital Twin", layout="wide")
st.title("Arjas Steel - Blast Furnace Digital Twin Dashboard")

model = BlastFurnaceSimulator()

# Sidebar: File uploader and controls
uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample (simulated) data", value=True if not uploaded_file else False)

# Select aggregation level for OPTIONAL aggregated views
agg_level = st.sidebar.selectbox(
    "Consolidation Level (for aggregation table below)",
    ["Hourly", "Daily", "Monthly", "Yearly"],
    index=0
)

# Number of last points to show (raw)
max_points = st.sidebar.slider("Show last N datapoints (raw)", min_value=10, max_value=200, value=50, step=10)

# Data acquisition logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
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
    if 'sensor_history' not in st.session_state:
        st.session_state['sensor_history'] = []
    history = st.session_state['sensor_history']
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

col1, col2 = st.columns([2, 1])

def aggregate_df(df, level):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    df = df.set_index('timestamp')
    if level == "Hourly":
        grouped = df.resample('H').mean(numeric_only=True)
    elif level == "Daily":
        grouped = df.resample('D').mean(numeric_only=True)
    elif level == "Monthly":
        grouped = df.resample('M').mean(numeric_only=True)
    elif level == "Yearly":
        grouped = df.resample('Y').mean(numeric_only=True)
    else:
        grouped = df
    grouped = grouped.reset_index()
    return grouped

def add_predictions_and_anomalies(df):
    df = df.copy()
    df['predicted_hot_metal'] = [
        model.simulate_step(row.to_dict())['predicted_hot_metal'] for _, row in df.iterrows()
    ]
    df['anomaly'] = [
        bool(detect_anomaly(row.to_dict())) for _, row in df.iterrows()
    ]
    return df

with col1:
    if not uploaded_file:
        if st.button("Get Latest Sensor Data"):
            raw = get_sensor_data()
            clean = clean_data(raw)
            st.session_state['sensor_history'].append(clean)
            st.success("New sensor data ingested!")
            history = st.session_state['sensor_history']

    if history:
        # -- RAW DATA, last N points --
        df = to_dataframe(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_last = df.tail(max_points)
        df_last = add_predictions_and_anomalies(df_last)

        st.subheader(f"Last {max_points} Raw Data Points with Model Prediction & Anomaly")
        st.dataframe(df_last)

        # Time series trends for all major parameters (raw data, last N)
        st.subheader("Time Series Trends (Last N points – Raw Data)")
        plot_cols = ['temperature', 'pressure', 'CO_content', 'feed_rate', 'air_flow', 'hot_metal_level', 'slag_rate']
        plot_cols = [col for col in plot_cols if col in df_last.columns]
        if plot_cols:
            st.line_chart(df_last.set_index('timestamp')[plot_cols], use_container_width=True)
        else:
            st.info("No process variables available for plotting.")

        # Predicted hot metal output time series (raw data, last N)
        st.subheader("Predicted Hot Metal Output (Time Series, Last N Raw Points)")
        if 'predicted_hot_metal' in df_last.columns:
            st.line_chart(df_last.set_index('timestamp')[['predicted_hot_metal']], use_container_width=True)
        else:
            st.info("No predicted hot metal output data available.")

        # Highlight anomalies (raw data, last N)
        anomaly_points = df_last[df_last['anomaly']]
        if not anomaly_points.empty:
            st.warning(f"Anomalies detected at {len(anomaly_points)} time points (raw data). See table below.")
            st.dataframe(anomaly_points)
        else:
            st.success("No anomalies detected in last N points (raw data).")

        # -- OPTIONAL: Aggregated data view below --
        st.subheader(f"Aggregated ({agg_level}) Data Table (for reference, not for time series plots)")
        agg_df = aggregate_df(df, agg_level)
        agg_df = add_predictions_and_anomalies(agg_df)
        st.dataframe(agg_df.tail(20))

    else:
        st.info("Click the button to get sensor data or upload a CSV.")

with col2:
    if history:
        latest = history[-1]
        # Simulate for latest only (for what-if etc)
        pred = model.simulate_step(latest)
        st.metric("Predicted Hot Metal Output (Latest)", f"{pred['predicted_hot_metal']:.2f}")

        # Anomaly detection
        anomaly = detect_anomaly(latest)
        if anomaly:
            st.error(f"Anomalies detected: {anomaly}")
        else:
            st.success("No anomalies detected.")

    # What-If Analysis section
    st.markdown("---")
    st.header("What-If Analysis")
    if history:
        what_if_interface(model, latest)
    else:
        st.info("Upload data or collect a sample to run What-If analysis.")

st.markdown("---")
st.caption("© Arjas Steel Digital Twin Example")
