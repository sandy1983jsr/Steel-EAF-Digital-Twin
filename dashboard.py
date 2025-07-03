import streamlit as st
import pandas as pd
from data_acquisition import get_sensor_data
from data_processing import clean_data, to_dataframe
from blast_furnace_model import BlastFurnaceSimulator, detect_anomaly

st.set_page_config(page_title="Arjas Steel Blast Furnace Digital Twin", layout="wide")
st.title("Arjas Steel - Blast Furnace Digital Twin Dashboard")

model = BlastFurnaceSimulator()

if 'sensor_history' not in st.session_state:
    st.session_state['sensor_history'] = []

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Get Latest Sensor Data"):
        raw = get_sensor_data()
        clean = clean_data(raw)
        st.session_state['sensor_history'].append(clean)
        st.success("New sensor data ingested!")

    if st.session_state['sensor_history']:
        df = to_dataframe(st.session_state['sensor_history'])
        st.dataframe(df.tail(20))

        # Show trends
        st.line_chart(df.set_index('timestamp')[['temperature', 'pressure', 'CO_content']])
        st.line_chart(df.set_index('timestamp')[['feed_rate', 'air_flow', 'hot_metal_level', 'slag_rate']])
    else:
        st.info("Click the button to get sensor data.")

with col2:
    if st.session_state['sensor_history']:
        latest = st.session_state['sensor_history'][-1]
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
