import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ---- Simulated SMS Model and Anomaly Detection ----

class SteelMeltingShopSimulator:
    def __init__(self):
        # Example model parameters, you can adapt with real logic
        self.last_predicted_heat = 0

    def simulate_step(self, row):
        # Simple digital twin logic for prediction, replace with your own!
        # Example: target_tapping_temp = function(alloy, power, oxygen, ...)
        predicted_tapping_temp = (
            0.4 * row.get("scrap_charge", 0)
            + 0.3 * row.get("power_consumption", 0)
            + 0.2 * row.get("oxygen_injection", 0)
            + 0.1 * row.get("lime_addition", 0)
        ) / 10 + 1500  # Dummy formula for illustration
        self.last_predicted_heat = predicted_tapping_temp
        return {"predicted_tapping_temp": predicted_tapping_temp}

def detect_sms_anomaly(row):
    # Example anomaly logic: flag if actual_tapping_temp deviates from predicted by > 30C
    predicted = row.get("predicted_tapping_temp", None)
    actual = row.get("actual_tapping_temp", None)
    if predicted is not None and actual is not None:
        if abs(predicted - actual) > 30:
            return f"Tapping Temp Deviation: {actual:.1f} vs Predicted {predicted:.1f}"
    return None

# ---- What-If Analysis Interface ----
def sms_what_if_interface(model, latest):
    st.subheader("What-If Analysis (Steel Melting Shop)")
    st.write("Adjust process parameters to see predicted tapping temperature.")
    # Simple example, expand with your SMS parameters!
    scrap_charge = st.number_input("Scrap Charge (tons)", value=float(latest.get("scrap_charge", 0)))
    power = st.number_input("Power Consumption (MWh)", value=float(latest.get("power_consumption", 0)))
    oxygen = st.number_input("Oxygen Injection (Nm3)", value=float(latest.get("oxygen_injection", 0)))
    lime = st.number_input("Lime Addition (tons)", value=float(latest.get("lime_addition", 0)))
    process_row = dict(
        scrap_charge=scrap_charge,
        power_consumption=power,
        oxygen_injection=oxygen,
        lime_addition=lime,
    )
    result = model.simulate_step(process_row)
    st.metric("Predicted Tapping Temp (°C)", f"{result['predicted_tapping_temp']:.1f}")

# ---- Data Processing & Streamlit App ----

st.set_page_config(page_title="Steel Melting Shop Digital Twin", layout="wide")
st.title("Steel Melting Shop - Digital Twin Dashboard")

sms_model = SteelMeltingShopSimulator()

uploaded_file = st.sidebar.file_uploader("Upload SMS CSV data", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample (simulated) data", value=True if not uploaded_file else False)

view_level = st.sidebar.selectbox(
    "View Time Series As",
    ["Raw Data", "Hourly", "Daily", "Monthly"],
    index=0
)
max_points = st.sidebar.slider("Show last N datapoints", min_value=10, max_value=200, value=50, step=10)

# ---- Data Acquisition ----

def create_sms_sample_data(n=200):
    # Generate 200 hourly records for SMS with random but realistic values
    start = datetime(2025, 6, 25)
    records = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        rec = dict(
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            scrap_charge=round(80 + 10 * (i%5) + (i%3)*2, 1),
            power_consumption=round(600 + (i%7)*5 + (i%2)*10, 1),
            oxygen_injection=round(1200 + (i%11)*6, 1),
            lime_addition=round(7 + (i%4)*0.3, 2),
            actual_tapping_temp=round(1550 + (i%5)*2 + (i%4)*0.5, 1),
            slag_basicity=round(3 + (i%4)*0.1, 2),
        )
        records.append(rec)
    return pd.DataFrame(records)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')  # Clean header
    expected_cols = set(["timestamp", "scrap_charge", "power_consumption", "oxygen_injection", "lime_addition", "actual_tapping_temp", "slag_basicity"])
    uploaded_cols = set(df.columns)
    if not expected_cols.issubset(uploaded_cols):
        st.warning(
            f"Missing columns in uploaded CSV. Expected: {expected_cols}\nFound: {uploaded_cols}"
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])
elif use_sample or not uploaded_file:
    if 'sms_history' not in st.session_state:
        st.session_state['sms_history'] = create_sms_sample_data(200)
    df = st.session_state['sms_history'].copy()
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

# ---- Resampling (Aggregation) ----
def resample_sms_data(df, period):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    df = df.set_index('timestamp')
    agg = df.resample(period).mean(numeric_only=True)
    agg = agg.reset_index()
    return agg

# ---- Model Prediction & Anomaly Detection ----
def add_sms_predictions_and_anomalies(df):
    df = df.copy()
    df['predicted_tapping_temp'] = [
        sms_model.simulate_step(row.to_dict())['predicted_tapping_temp'] for _, row in df.iterrows()
    ]
    df['anomaly'] = [
        detect_sms_anomaly(row.to_dict()) for _, row in df.iterrows()
    ]
    return df

col1, col2 = st.columns([2, 1])

with col1:
    if not uploaded_file and st.button("Get New Sample Data"):
        st.session_state['sms_history'] = create_sms_sample_data(200)
        st.success("New sample data generated!")
        df = st.session_state['sms_history']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")

    # Choose view (raw or resampled)
    if view_level == "Raw Data":
        df_view = df.tail(max_points).copy()
    elif view_level == "Hourly":
        df_resampled = resample_sms_data(df, "H")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Daily":
        df_resampled = resample_sms_data(df, "D")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Monthly":
        df_resampled = resample_sms_data(df, "M")
        df_view = df_resampled.tail(max_points).copy()
    else:
        df_view = df.tail(max_points).copy()

    # Add model predictions and anomalies
    df_view = add_sms_predictions_and_anomalies(df_view)

    st.subheader(f"Time Series Table ({view_level}, last {max_points} points)")
    st.dataframe(df_view)

    plot_cols = ['scrap_charge', 'power_consumption', 'oxygen_injection', 'lime_addition', 'actual_tapping_temp', 'slag_basicity']
    plot_cols = [col for col in plot_cols if col in df_view.columns]

    st.subheader(f"Time Series Trends ({view_level}, last {max_points} points)")
    if plot_cols:
        st.line_chart(df_view.set_index('timestamp')[plot_cols])
    else:
        st.info("No process variables available for plotting.")

    st.subheader(f"Predicted Tapping Temp ({view_level}, last {max_points} points)")
    if 'predicted_tapping_temp' in df_view.columns:
        st.line_chart(df_view.set_index('timestamp')[['predicted_tapping_temp']])
    else:
        st.info("No predicted tapping temp data available.")

    anomaly_points = df_view[df_view['anomaly'].notna() & (df_view['anomaly'] != "")]
    if not anomaly_points.empty:
        st.warning(f"Anomalies detected at {len(anomaly_points)} time points. See table below.")
        st.dataframe(anomaly_points)
    else:
        st.success(f"No anomalies detected in last {max_points} points ({view_level}).")

with col2:
    latest = df.iloc[-1].to_dict()
    pred = sms_model.simulate_step(latest)
    st.metric("Predicted Tapping Temp (Latest)", f"{pred['predicted_tapping_temp']:.1f} °C")
    anomaly = detect_sms_anomaly({**latest, **pred})
    if anomaly:
        st.error(f"Anomalies detected: {anomaly}")
    else:
        st.success("No anomalies detected.")

    st.markdown("---")
    st.header("What-If Analysis")
    sms_what_if_interface(sms_model, latest)

st.markdown("---")
st.caption("© Steel Melting Shop Digital Twin Example")
