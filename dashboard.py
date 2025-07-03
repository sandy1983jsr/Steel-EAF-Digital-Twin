import streamlit as st
import pandas as pd
from data_acquisition import get_sensor_data
from data_processing import clean_data, to_dataframe
from blast_furnace_model import BlastFurnaceSimulator, detect_anomaly
from what_if_analysis import what_if_interface

st.set_page_config(page_title="Arjas Steel Blast Furnace Digital Twin", layout="wide")
st.title("Arjas Steel - Blast Furnace Digital Twin Dashboard")

model = BlastFurnaceSimulator()

uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample (simulated) data", value=True if not uploaded_file else False)

view_level = st.sidebar.selectbox(
    "View Time Series As",
    ["Raw Data", "Hourly", "Daily", "Monthly"],
    index=0
)
max_points = st.sidebar.slider("Show last N datapoints", min_value=10, max_value=200, value=50, step=10)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    expected_cols = {"timestamp", "temperature", "pressure", "CO_content", "feed_rate", "air_flow", "hot_metal_level", "slag_rate"}
    uploaded_cols = set(df.columns)
    if not expected_cols.issubset(uploaded_cols):
        st.warning(
            f"Some expected columns are missing from the uploaded CSV.\n"
            f"Expected: {expected_cols}\nFound: {uploaded_cols}"
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])
    history = df.to_dict("records")
elif use_sample or not uploaded_file:
    if 'sensor_history' not in st.session_state:
        st.session_state['sensor_history'] = []
    history = st.session_state['sensor_history']
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

def add_predictions_and_anomalies(df):
    df = df.copy()
    df['predicted_hot_metal'] = [
        model.simulate_step(row.to_dict())['predicted_hot_metal'] for _, row in df.iterrows()
    ]
    df['anomaly'] = [
        bool(detect_anomaly(row.to_dict())) for _, row in df.iterrows()
    ]
    return df

def resample_data(df, period):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    df = df.set_index('timestamp')
    agg = df.resample(period).mean(numeric_only=True)
    agg = agg.reset_index()
    return agg

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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        # Always take last max_points AFTER resampling (or for raw)
        if view_level == "Raw Data":
            df_view = df.tail(max_points).copy()
        elif view_level == "Hourly":
            df_resampled = resample_data(df, "H")
            df_view = df_resampled.tail(max_points).copy()
        elif view_level == "Daily":
            df_resampled = resample_data(df, "D")
            df_view = df_resampled.tail(max_points).copy()
        elif view_level == "Monthly":
            df_resampled = resample_data(df, "M")
            df_view = df_resampled.tail(max_points).copy()
        else:
            df_view = df.tail(max_points).copy()

        # Use index as serial number for x axis if all timestamps collapse to same value
        if len(df_view['timestamp'].unique()) == 1:
            df_view['serial'] = range(len(df_view))
            x_axis = 'serial'
            st.warning("All timestamps are the same. Showing serial number as x-axis.")
        else:
            x_axis = 'timestamp'

        # Add model predictions and anomalies
        df_view = add_predictions_and_anomalies(df_view)

        st.subheader(f"Time Series Table ({view_level}, last {max_points} points)")
        st.dataframe(df_view)

        plot_cols = ['temperature', 'pressure', 'CO_content', 'feed_rate', 'air_flow', 'hot_metal_level', 'slag_rate']
        plot_cols = [col for col in plot_cols if col in df_view.columns]

        st.subheader(f"Time Series Trends ({view_level}, last {max_points} points)")
        if plot_cols:
            st.line_chart(df_view.set_index(x_axis)[plot_cols])
        else:
            st.info("No process variables available for plotting.")

        st.subheader(f"Predicted Hot Metal Output ({view_level}, last {max_points} points)")
        if 'predicted_hot_metal' in df_view.columns:
            st.line_chart(df_view.set_index(x_axis)[['predicted_hot_metal']])
        else:
            st.info("No predicted hot metal output data available.")

        anomaly_points = df_view[df_view['anomaly']]
        if not anomaly_points.empty:
            st.warning(f"Anomalies detected at {len(anomaly_points)} time points. See table below.")
            st.dataframe(anomaly_points)
        else:
            st.success(f"No anomalies detected in last {max_points} points ({view_level}).")
    else:
        st.info("Click the button to get sensor data or upload a CSV.")

with col2:
    if history:
        df = to_dataframe(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")
        latest = df.iloc[-1].to_dict()
        pred = model.simulate_step(latest)
        st.metric("Predicted Hot Metal Output (Latest)", f"{pred['predicted_hot_metal']:.2f}")
        anomaly = detect_anomaly(latest)
        if anomaly:
            st.error(f"Anomalies detected: {anomaly}")
        else:
            st.success("No anomalies detected.")

    st.markdown("---")
    st.header("What-If Analysis")
    if history:
        what_if_interface(model, latest)
    else:
        st.info("Upload data or collect a sample to run What-If analysis.")

st.markdown("---")
st.caption("© Arjas Steel Digital Twin Example")
