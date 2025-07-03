import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Advanced SMS Digital Twin Model & KPIs ---

class AdvancedSteelMeltingShopSimulator:
    def __init__(self):
        pass

    def simulate_step(self, row):
        # Example physically-inspired SMS twin logic (replace coefficients as needed)
        scrap = row.get("scrap_charge", 0)
        dri = row.get("dri_charge", 0)
        hot_metal = row.get("hot_metal_charge", 0)
        oxygen = row.get("oxygen_injection", 0)
        power = row.get("power_consumption", 0)
        lime = row.get("lime_addition", 0)
        dolomite = row.get("dolomite_addition", 0)
        alloy = row.get("alloy_addition", 0)
        total_charge = scrap + dri + hot_metal

        # 1. Predicted tapping temperature (°C)
        predicted_tapping_temp = (
            0.32 * hot_metal +
            0.25 * scrap +
            0.22 * dri +
            0.09 * lime +
            0.04 * dolomite +
            0.02 * alloy +
            0.08 * oxygen +
            0.12 * power
        ) / (total_charge + 1e-6) + 1470 + np.random.normal(0, 2)

        # 2. Predicted heat time (min)
        predicted_heat_time = (
            50 +
            0.06 * total_charge +
            0.018 * power -
            0.015 * hot_metal +
            0.01 * oxygen
        ) + np.random.normal(0, 0.5)

        # 3. Predicted yield (%)
        predicted_yield = (
            94.5 +
            0.01 * hot_metal -
            0.005 * scrap +
            0.004 * dri -
            0.002 * alloy
        ) + np.random.normal(0, 0.1)

        # 4. Predicted slag basicity (CaO/SiO2)
        predicted_basicity = (
            2.8 +
            0.03 * lime -
            0.02 * dolomite +
            0.01 * (hot_metal + scrap) / (total_charge + 1e-6)
        ) + np.random.normal(0, 0.01)

        # 5. Predicted power (MWh)
        predicted_power = (
            550 +
            0.9 * (scrap + dri) +
            0.35 * oxygen +
            0.04 * lime
        ) + np.random.normal(0, 5)

        # 6. Predicted alloy addition (kg/ton steel)
        predicted_alloy = (
            8.0 +
            0.35 * (scrap/total_charge if total_charge > 0 else 0) +
            0.1 * power / 1000
        ) + np.random.normal(0, 0.2)

        # 7. Predicted tap-to-tap time (min)
        predicted_tap_to_tap = predicted_heat_time + 7 + np.random.normal(0, 0.3)

        return {
            "predicted_tapping_temp": predicted_tapping_temp,
            "predicted_heat_time": predicted_heat_time,
            "predicted_yield": predicted_yield,
            "predicted_basicity": predicted_basicity,
            "predicted_power": predicted_power,
            "predicted_alloy": predicted_alloy,
            "predicted_tap_to_tap": predicted_tap_to_tap,
        }

def detect_sms_kpi_anomalies(row):
    anomalies = []
    # Tapping temp deviation
    if abs(row.get("predicted_tapping_temp", 0) - row.get("actual_tapping_temp", 0)) > 30:
        anomalies.append("Tapping Temp deviation")
    # Heat time high/low
    if "heat_time" in row and abs(row.get("predicted_heat_time", 0) - row.get("heat_time", 0)) > 10:
        anomalies.append("Heat Time deviation")
    # Yield low
    if "yield_percent" in row and (row.get("yield_percent", 0) < 93 or row.get("yield_percent", 0) > 98):
        anomalies.append("Yield out of bounds")
    # Basicity out of bounds
    if "slag_basicity" in row and not (2.7 <= row.get("slag_basicity", 0) <= 3.3):
        anomalies.append("Basicity out of bounds")
    # Power deviation
    if "power_consumption" in row and abs(row.get("predicted_power", 0) - row.get("power_consumption", 0)) > 60:
        anomalies.append("Power deviation")
    # Alloy deviation
    if "alloy_addition" in row and abs(row.get("predicted_alloy", 0) - row.get("alloy_addition", 0)) > 2:
        anomalies.append("Alloy addition deviation")
    # Tap-to-tap time deviation
    if "tap_to_tap_time" in row and abs(row.get("predicted_tap_to_tap", 0) - row.get("tap_to_tap_time", 0)) > 10:
        anomalies.append("Tap-to-Tap deviation")
    return ", ".join(anomalies) if anomalies else None

# --- What-If Analysis Extended ---
def sms_what_if_interface_advanced(model, latest):
    st.subheader("What-If Analysis (Steel Melting Shop, Advanced)")
    st.write("Adjust process parameters to see predicted KPIs.")
    scrap_charge = st.number_input("Scrap Charge (tons)", value=float(latest.get("scrap_charge", 80)))
    dri_charge = st.number_input("DRI Charge (tons)", value=float(latest.get("dri_charge", 20)))
    hot_metal_charge = st.number_input("Hot Metal Charge (tons)", value=float(latest.get("hot_metal_charge", 50)))
    power = st.number_input("Power Consumption (MWh)", value=float(latest.get("power_consumption", 600)))
    oxygen = st.number_input("Oxygen Injection (Nm3)", value=float(latest.get("oxygen_injection", 1200)))
    lime = st.number_input("Lime Addition (tons)", value=float(latest.get("lime_addition", 7)))
    dolomite = st.number_input("Dolomite Addition (tons)", value=float(latest.get("dolomite_addition", 2)))
    alloy = st.number_input("Alloy Addition (tons)", value=float(latest.get("alloy_addition", 1)))
    process_row = dict(
        scrap_charge=scrap_charge,
        dri_charge=dri_charge,
        hot_metal_charge=hot_metal_charge,
        power_consumption=power,
        oxygen_injection=oxygen,
        lime_addition=lime,
        dolomite_addition=dolomite,
        alloy_addition=alloy,
    )
    result = model.simulate_step(process_row)
    st.metric("Predicted Tapping Temp (°C)", f"{result['predicted_tapping_temp']:.1f}")
    st.metric("Predicted Heat Time (min)", f"{result['predicted_heat_time']:.1f}")
    st.metric("Predicted Yield (%)", f"{result['predicted_yield']:.2f}")
    st.metric("Predicted Slag Basicity", f"{result['predicted_basicity']:.2f}")
    st.metric("Predicted Power (MWh)", f"{result['predicted_power']:.1f}")
    st.metric("Predicted Alloy (t)", f"{result['predicted_alloy']:.2f}")
    st.metric("Predicted Tap-to-Tap (min)", f"{result['predicted_tap_to_tap']:.1f}")

# ---- Sample Data Generation with All KPIs ----
def create_sms_sample_data_advanced(n=200):
    start = datetime(2025, 6, 25)
    records = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        scrap = np.random.uniform(70, 95)
        dri = np.random.uniform(15, 30)
        hot_metal = np.random.uniform(40, 65)
        power = np.random.uniform(580, 650)
        oxygen = np.random.uniform(1150, 1300)
        lime = np.random.uniform(6, 10)
        dolomite = np.random.uniform(1.2, 3.0)
        alloy = np.random.uniform(0.8, 1.5)
        # Model predictions
        sim = AdvancedSteelMeltingShopSimulator()
        pred = sim.simulate_step(dict(
            scrap_charge=scrap,
            dri_charge=dri,
            hot_metal_charge=hot_metal,
            power_consumption=power,
            oxygen_injection=oxygen,
            lime_addition=lime,
            dolomite_addition=dolomite,
            alloy_addition=alloy,
        ))
        # Add some noise for "actual" values
        actual_tapping_temp = pred['predicted_tapping_temp'] + np.random.normal(0, 10)
        heat_time = pred['predicted_heat_time'] + np.random.normal(0, 5)
        yield_percent = pred['predicted_yield'] + np.random.normal(0, 1)
        slag_basicity = pred['predicted_basicity'] + np.random.normal(0, 0.03)
        tap_to_tap_time = pred['predicted_tap_to_tap'] + np.random.normal(0, 2)
        records.append(dict(
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            scrap_charge=scrap,
            dri_charge=dri,
            hot_metal_charge=hot_metal,
            power_consumption=power,
            oxygen_injection=oxygen,
            lime_addition=lime,
            dolomite_addition=dolomite,
            alloy_addition=alloy,
            actual_tapping_temp=actual_tapping_temp,
            heat_time=heat_time,
            yield_percent=yield_percent,
            slag_basicity=slag_basicity,
            tap_to_tap_time=tap_to_tap_time,
        ))
    return pd.DataFrame(records)

# ---- Data Acquisition & App ----

st.set_page_config(page_title="Steel Melting Shop Advanced Digital Twin", layout="wide")
st.title("Steel Melting Shop - Advanced Digital Twin Dashboard")

sms_model = AdvancedSteelMeltingShopSimulator()

uploaded_file = st.sidebar.file_uploader("Upload SMS CSV data", type=["csv"])
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
        "timestamp", "scrap_charge", "dri_charge", "hot_metal_charge",
        "power_consumption", "oxygen_injection", "lime_addition", "dolomite_addition", "alloy_addition",
        "actual_tapping_temp", "heat_time", "yield_percent", "slag_basicity", "tap_to_tap_time"
    ])
    uploaded_cols = set(df.columns)
    if not expected_cols.issubset(uploaded_cols):
        st.warning(
            f"Missing columns in uploaded CSV. Expected: {expected_cols}\nFound: {uploaded_cols}"
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])
elif use_sample or not uploaded_file:
    if 'sms_history' not in st.session_state:
        st.session_state['sms_history'] = create_sms_sample_data_advanced(200)
    df = st.session_state['sms_history'].copy()
else:
    st.error("Please upload a valid CSV file or select 'Use sample data'.")
    st.stop()

def resample_sms_data_advanced(df, period):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    df = df.set_index('timestamp')
    agg = df.resample(period).mean(numeric_only=True)
    agg = agg.reset_index()
    return agg

def add_sms_predictions_and_anomalies_advanced(df):
    df = df.copy()
    preds = [sms_model.simulate_step(row.to_dict()) for _, row in df.iterrows()]
    for kpi in preds[0].keys():
        df[f'predicted_{kpi.replace("predicted_", "")}'] = [p[kpi] for p in preds]
    df['anomaly'] = [
        detect_sms_kpi_anomalies({**row._asdict(), **preds[i]}) for i, row in enumerate(df.itertuples(index=False))
    ]
    return df

col1, col2 = st.columns([2, 1])

with col1:
    if not uploaded_file and st.button("Get New Sample Data"):
        st.session_state['sms_history'] = create_sms_sample_data_advanced(200)
        st.success("New sample data generated!")
        df = st.session_state['sms_history']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")

    if view_level == "Raw Data":
        df_view = df.tail(max_points).copy()
    elif view_level == "Hourly":
        df_resampled = resample_sms_data_advanced(df, "H")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Daily":
        df_resampled = resample_sms_data_advanced(df, "D")
        df_view = df_resampled.tail(max_points).copy()
    elif view_level == "Monthly":
        df_resampled = resample_sms_data_advanced(df, "M")
        df_view = df_resampled.tail(max_points).copy()
    else:
        df_view = df.tail(max_points).copy()

    df_view = add_sms_predictions_and_anomalies_advanced(df_view)

    st.subheader(f"Time Series Table ({view_level}, last {max_points} points)")
    st.dataframe(df_view)

    kpi_plot_cols = [
        'actual_tapping_temp', 'heat_time', 'yield_percent', 'slag_basicity',
        'tap_to_tap_time', 'power_consumption', 'scrap_charge', 'dri_charge', 'hot_metal_charge',
        'oxygen_injection', 'lime_addition', 'dolomite_addition', 'alloy_addition'
    ]
    plot_cols = [col for col in kpi_plot_cols if col in df_view.columns]

    st.subheader(f"SMS KPI Trends ({view_level}, last {max_points} points)")
    if plot_cols:
        st.line_chart(df_view.set_index('timestamp')[plot_cols])
    else:
        st.info("No process variables available for plotting.")

    # Predicted vs Actual for major KPIs
    st.subheader(f"Predicted vs Actual KPIs ({view_level}, last {max_points} points)")
    for kpi in ["tapping_temp", "heat_time", "yield", "basicity", "power", "alloy", "tap_to_tap"]:
        pred_col = f"predicted_{kpi}"
        actual_col = None
        if kpi == "tapping_temp":
            actual_col = "actual_tapping_temp"
        elif kpi == "heat_time":
            actual_col = "heat_time"
        elif kpi == "yield":
            actual_col = "yield_percent"
        elif kpi == "basicity":
            actual_col = "slag_basicity"
        elif kpi == "tap_to_tap":
            actual_col = "tap_to_tap_time"
        elif kpi == "power":
            actual_col = "power_consumption"
        elif kpi == "alloy":
            actual_col = "alloy_addition"
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
    pred = sms_model.simulate_step(latest)
    st.metric("Predicted Tapping Temp (Latest)", f"{pred['predicted_tapping_temp']:.1f} °C")
    st.metric("Predicted Heat Time (Latest)", f"{pred['predicted_heat_time']:.1f} min")
    st.metric("Predicted Yield (Latest)", f"{pred['predicted_yield']:.2f} %")
    st.metric("Predicted Basicity (Latest)", f"{pred['predicted_basicity']:.2f}")
    st.metric("Predicted Power (Latest)", f"{pred['predicted_power']:.1f} MWh")
    st.metric("Predicted Alloy (Latest)", f"{pred['predicted_alloy']:.2f} t")
    st.metric("Predicted Tap-to-Tap (Latest)", f"{pred['predicted_tap_to_tap']:.1f} min")
    anomaly = detect_sms_kpi_anomalies({**latest, **pred})
    if anomaly:
        st.error(f"Anomalies detected: {anomaly}")
    else:
        st.success("No anomalies detected.")

    st.markdown("---")
    st.header("What-If Analysis")
    sms_what_if_interface_advanced(sms_model, latest)

st.markdown("---")
st.caption("© Steel Melting Shop Advanced Digital Twin Example")
