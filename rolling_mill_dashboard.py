import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Advanced Rolling Mill Digital Twin Model & KPIs ---

class RollingMillSimulator:
    def __init__(self):
        pass

    def simulate_step(self, row):
        # Inputs
        entry_temp = row.get("entry_temp", 0)
        thickness_in = row.get("thickness_in", 0)
        thickness_out = row.get("thickness_out", 0)
        width = row.get("width", 0)
        speed_setpoint = row.get("speed_setpoint", 0)
        rolling_power = row.get("rolling_power", 0)
        tension = row.get("tension", 0)
        alloy_grade = row.get("alloy_grade", 1.0)  # e.g., 1.0 for standard, >1 for harder grades

        # Predicted Exit Temperature (°C)
        predicted_exit_temp = (
            entry_temp
            - 20 * (thickness_in - thickness_out)
            - 0.1 * speed_setpoint
            - 5 * (alloy_grade - 1)
            + np.random.normal(0, 2)
        )

        # Predicted Rolling Force (kN)
        predicted_rolling_force = (
            1000 * (thickness_in - thickness_out)
            * width
            * alloy_grade
            / (thickness_in + 1e-6)
            + 0.05 * tension
            + np.random.normal(0, 50)
        )

        # Predicted Mill Speed (m/s)
        predicted_speed = (
            speed_setpoint
            + 0.1 * (entry_temp - predicted_exit_temp)
            - 0.02 * rolling_power
            + np.random.normal(0, 0.1)
        )

        # Predicted Yield (%)
        predicted_yield = (
            98.5
            - 0.03 * (alloy_grade - 1) * 100
            - 0.01 * abs(thickness_in - thickness_out)
            + np.random.normal(0, 0.05)
        )

        # Predicted Power (kWh)
        predicted_power = (
            0.8 * predicted_rolling_force * predicted_speed / 100
            + 10 * alloy_grade
            + np.random.normal(0, 5)
        )

        # Predicted Surface Defects (count)
        predicted_surface_defects = (
            0.05 * (alloy_grade - 1) * 100
            + 0.01 * abs(entry_temp - predicted_exit_temp)
            + np.random.normal(0, 0.3)
        )

        return {
            "predicted_exit_temp": predicted_exit_temp,
            "predicted_rolling_force": predicted_rolling_force,
            "predicted_speed": predicted_speed,
            "predicted_yield": predicted_yield,
            "predicted_power": predicted_power,
            "predicted_surface_defects": max(0, predicted_surface_defects),
        }

def detect_rolling_anomalies(row):
    anomalies = []
    # Exit temp deviation
    if abs(row.get("predicted_exit_temp", 0) - row.get("exit_temp", 0)) > 25:
        anomalies.append("Exit Temp deviation")
    # Rolling force deviation
    if abs(row.get("predicted_rolling_force", 0) - row.get("rolling_force", 0)) > 200:
        anomalies.append("Rolling Force deviation")
    # Yield low
    if "yield_percent" in row and (row.get("yield_percent", 0) < 96 or row.get("yield_percent", 0) > 99.5):
        anomalies.append("Yield out of bounds")
    # Power deviation
    if "rolling_power" in row and abs(row.get("predicted_power", 0) - row.get("rolling_power", 0)) > 20:
        anomalies.append("Power deviation")
    # Surface defects high
    if "surface_defects" in row and row.get("surface_defects", 0) > 2:
        anomalies.append("Surface defects high")
    return ", ".join(anomalies) if anomalies else None

# --- What-If Analysis ---
def rolling_what_if_interface(model, latest):
    st.subheader("What-If Analysis (Rolling Mill)")
    st.write("Adjust process parameters to see predicted KPIs.")
    entry_temp = st.number_input("Entry Temperature (°C)", value=float(latest.get("entry_temp", 1100)))
    thickness_in = st.number_input("Thickness In (mm)", value=float(latest.get("thickness_in", 200)))
    thickness_out = st.number_input("Thickness Out (mm)", value=float(latest.get("thickness_out", 10)))
    width = st.number_input("Width (mm)", value=float(latest.get("width", 1200)))
    speed_setpoint = st.number_input("Mill Speed Setpoint (m/s)", value=float(latest.get("speed_setpoint", 2.5)))
    rolling_power = st.number_input("Rolling Power (kWh)", value=float(latest.get("rolling_power", 350)))
    tension = st.number_input("Tension (kN)", value=float(latest.get("tension", 50)))
    alloy_grade = st.number_input("Alloy Grade Factor", value=float(latest.get("alloy_grade", 1.0)))
    process_row = dict(
        entry_temp=entry_temp,
        thickness_in=thickness_in,
        thickness_out=thickness_out,
        width=width,
        speed_setpoint=speed_setpoint,
        rolling_power=rolling_power,
        tension=tension,
        alloy_grade=alloy_grade,
    )
    result = model.simulate_step(process_row)
    st.metric("Predicted Exit Temp (°C)", f"{result['predicted_exit_temp']:.1f}")
    st.metric("Predicted Rolling Force (kN)", f"{result['predicted_rolling_force']:.1f}")
    st.metric("Predicted Mill Speed (m/s)", f"{result['predicted_speed']:.2f}")
    st.metric("Predicted Yield (%)", f"{result['predicted_yield']:.2f}")
    st.metric("Predicted Power (kWh)", f"{result['predicted_power']:.1f}")
    st.metric("Predicted Surface Defects (count)", f"{result['predicted_surface_defects']:.2f}")

# ---- Sample Data Generation ----
def create_rolling_sample_data(n=200):
    start = datetime(2025, 6, 25)
    records = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        entry_temp = np.random.uniform(1050, 1200)
        thickness_in = np.random.uniform(180, 220)
        thickness_out = np.random.uniform(7, 15)
        width = np.random.uniform(1100, 1350)
        speed_setpoint = np.random.uniform(2.1, 3.0)
        rolling_power = np.random.uniform(320, 400)
        tension = np.random.uniform(40, 60)
        alloy_grade = np.random.choice([1.0, 1.1, 1.2, 1.3], p=[0.5, 0.25, 0.15, 0.1])
        # Model predictions
        sim = RollingMillSimulator()
        pred = sim.simulate_step(dict(
            entry_temp=entry_temp,
            thickness_in=thickness_in,
            thickness_out=thickness_out,
            width=width,
            speed_setpoint=speed_setpoint,
            rolling_power=rolling_power,
            tension=tension,
            alloy_grade=alloy_grade,
        ))
        # Add some noise for "actual" values
        exit_temp = pred['predicted_exit_temp'] + np.random.normal(0, 8)
        rolling_force = pred['predicted_rolling_force'] + np.random.normal(0, 60)
        speed = pred['predicted_speed'] + np.random.normal(0, 0.08)
        yield_percent = pred['predicted_yield'] + np.random.normal(0, 0.2)
        power = pred['predicted_power'] + np.random.normal(0, 9)
        surface_defects = max(0, pred['predicted_surface_defects'] + np.random.normal(0, 0.6))
        records.append(dict(
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            entry_temp=entry_temp,
            thickness_in=thickness_in,
            thickness_out=thickness_out,
            width=width,
            speed_setpoint=speed_setpoint,
            rolling_power=power,
            tension=tension,
            alloy_grade=alloy_grade,
            exit_temp=exit_temp,
            rolling_force=rolling_force,
            speed=speed,
            yield_percent=yield_percent,
            surface_defects=surface_defects,
        ))
    return pd.DataFrame(records)

# ---- Data Acquisition & App ----

st.set_page_config(page_title="Steel Rolling Mill Digital Twin", layout="wide")
st.title("Steel Rolling Mill - Advanced Digital Twin Dashboard")

rolling_model = RollingMillSimulator()

uploaded_file = st.sidebar.file_uploader("Upload Rolling Mill CSV data", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample (simulated) data", value=True if not uploaded_file else False)

view_level = st.sidebar.selectbox(
    "View Time Series As",
    ["Raw Data", "Hourly", "Daily", "Monthly"],
    index=0
)
max_points = st.sidebar.slider("Show last N datapoints", min_value=10, max_value=200, value=50, step=10)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    expected_cols = set([
        "timestamp", "entry_temp", "thickness_in", "thickness_out", "width",
        "speed_setpoint", "rolling_power", "tension", "alloy_grade",
        "exit_temp", "rolling_force", "speed", "yield_percent", "surface_defects"
    ])
    uploaded_cols = set(df.columns)
    if not expected_cols.issubset(uploaded_cols):
        st.warning(
            f"Missing columns in uploaded CSV. Expected: {expected_cols}\nFound: {uploaded_cols}"
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])
elif use_sample or not uploaded_file:
    if 'rolling_history' not in st.session_state:
        st.session_state['rolling_history'] = create_rolling_sample_data(200)
    df = st.session_state['rolling_history'].copy()
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

def resample_rolling_data(df, period):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    df = df.set_index('timestamp')
    agg = df.resample(period).mean(numeric_only=True)
    agg = agg.reset_index()
    return agg

def add_rolling_predictions_and_anomalies(df):
    df = df.copy()
    preds = [rolling_model.simulate_step(row.to_dict()) for _, row in df.iterrows()]
    for kpi in preds[0].keys():
        df[f'predicted_{kpi.replace("predicted_", "")}'] = [p[kpi] for p in preds]
    df['anomaly'] = [
        detect_rolling_anomalies({**row._asdict(), **preds[i]}) for i, row in enumerate(df.itertuples(index=False))
    ]
    return df

col1, col2 = st.columns([2, 1])

with col1:
    if not uploaded_file and st.button("Get New Sample Data"):
        st.session_state['rolling_history'] = create_rolling_sample_data(200)
        st.success("New sample data generated!")
        df = st.session_state['rolling_history']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")

    if view_level == "Raw Data":
        df_view = df.tail(max_points).copy()
    elif view_level == "Hourly":
        df_resampled = resample_rolling_data(df, "H")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Daily":
        df_resampled = resample_rolling_data(df, "D")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Monthly":
        df_resampled = resample_rolling_data(df, "M")
        df_view = df_resampled.tail(max_points).copy()
    else:
        df_view = df.tail(max_points).copy()

    df_view = add_rolling_predictions_and_anomalies(df_view)

    st.subheader(f"Time Series Table ({view_level}, last {max_points} points)")
    st.dataframe(df_view)

    kpi_plot_cols = [
        'entry_temp', 'exit_temp', 'thickness_in', 'thickness_out', 'width',
        'speed_setpoint', 'speed', 'rolling_power', 'tension', 'alloy_grade',
        'rolling_force', 'yield_percent', 'surface_defects'
    ]
    plot_cols = [col for col in kpi_plot_cols if col in df_view.columns]

    st.subheader(f"Rolling Mill KPI Trends ({view_level}, last {max_points} points)")
    if plot_cols:
        st.line_chart(df_view.set_index('timestamp')[plot_cols])
    else:
        st.info("No process variables available for plotting.")

    # Predicted vs Actual for major KPIs
    st.subheader(f"Predicted vs Actual KPIs ({view_level}, last {max_points} points)")
    for kpi in ["exit_temp", "rolling_force", "speed", "yield", "power", "surface_defects"]:
        pred_col = f"predicted_{kpi}"
        actual_col = None
        if kpi == "exit_temp":
            actual_col = "exit_temp"
        elif kpi == "rolling_force":
            actual_col = "rolling_force"
        elif kpi == "speed":
            actual_col = "speed"
        elif kpi == "yield":
            actual_col = "yield_percent"
        elif kpi == "power":
            actual_col = "rolling_power"
        elif kpi == "surface_defects":
            actual_col = "surface_defects"
        if pred_col in df_view.columns and actual_col in df_view.columns:
            st.line_chart(df_view.set_index('timestamp')[[actual_col, pred_col]])

    anomaly_points = df_view[df_view['anomaly'].notna() & (df_view['anomaly'] != "")]
    if not anomaly_points.empty:
        st.warning(f"Anomalies detected at {len(anomaly_points)} time points. See table below.")
        st.dataframe(anomaly_points)
    else:
        st.success(f"No anomalies detected in last {max_points} points ({view_level}).")

with col2:
    latest = df.iloc[-1].to_dict()
    pred = rolling_model.simulate_step(latest)
    st.metric("Predicted Exit Temp (Latest)", f"{pred['predicted_exit_temp']:.1f} °C")
    st.metric("Predicted Rolling Force (Latest)", f"{pred['predicted_rolling_force']:.1f} kN")
    st.metric("Predicted Mill Speed (Latest)", f"{pred['predicted_speed']:.2f} m/s")
    st.metric("Predicted Yield (Latest)", f"{pred['predicted_yield']:.2f} %")
    st.metric("Predicted Power (Latest)", f"{pred['predicted_power']:.1f} kWh")
    st.metric("Predicted Surface Defects (Latest)", f"{pred['predicted_surface_defects']:.2f}")
    anomaly = detect_rolling_anomalies({**latest, **pred})
    if anomaly:
        st.error(f"Anomalies detected: {anomaly}")
    else:
        st.success("No anomalies detected.")

    st.markdown("---")
    st.header("What-If Analysis")
    rolling_what_if_interface(rolling_model, latest)

st.markdown("---")
st.caption("© Steel Rolling Mill Advanced Digital Twin Example")
