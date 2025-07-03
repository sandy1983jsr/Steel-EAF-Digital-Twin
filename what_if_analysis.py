import streamlit as st

def what_if_interface(model, base_input):
    st.header("🔮 What-If Analysis Scenarios")

    scenario = st.selectbox(
        "Select What-If Scenario",
        [
            "Raw Material Quality Variations",
            "Fuel and Energy Adjustments",
            "Process Parameter Changes",
            "Operational Disturbances",
            "Slag and Hot Metal Chemistry",
            "Environmental and Energy Compliance",
            "Production Planning",
            "Equipment Failure or Maintenance"
        ]
    )

    # Clone base input for manipulation
    input_params = base_input.copy()

    if scenario == "Raw Material Quality Variations":
        st.subheader("Iron Ore Grade / Impurity Changes")
        input_params["CO_content"] = st.slider("CO Content (%)", 14.0, 25.0, float(input_params.get("CO_content", 20.0)), 0.1)
        input_params["feed_rate"] = st.slider("Feed Rate (TPH)", 30.0, 100.0, float(input_params.get("feed_rate", 60.0)), 1.0)
        st.caption("Simulate effect of Fe content drop (lower CO_content) or increased impurities (lower feed rate).")

    elif scenario == "Fuel and Energy Adjustments":
        st.subheader("Coke Rate / PCI / NG / Hydrogen")
        input_params["feed_rate"] = st.slider("Coke Rate (TPH)", 30.0, 100.0, float(input_params.get("feed_rate", 60.0)), 1.0)
        input_params["CO_content"] = st.slider("PCI/NG/H2 Injection Ratio (%)", 10.0, 30.0, float(input_params.get("CO_content", 20.0)), 0.1)
        st.caption("Increase/decrease coke or alternative fuel rates.")

    elif scenario == "Process Parameter Changes":
        st.subheader("Blast Temperature, Oxygen, Pressure")
        input_params["temperature"] = st.slider("Blast Temperature (°C)", 1100.0, 1700.0, float(input_params.get("temperature", 1400.0)), 5.0)
        input_params["pressure"] = st.slider("Blast Pressure (Bar)", 1.0, 6.0, float(input_params.get("pressure", 3.0)), 0.1)
        input_params["air_flow"] = st.slider("Blast Air Flow (Nm3/min)", 600.0, 1500.0, float(input_params.get("air_flow", 1000.0)), 10.0)
        st.caption("Test effect of changing blast temperature, pressure, and air flow.")

    elif scenario == "Operational Disturbances":
        st.subheader("Sudden Air Flow/Pressure/Delays/Blockage")
        disturbance = st.radio(
            "Disturbance Type",
            ["Drop in Air Flow", "Spike in Pressure", "Tapping Delay", "Tuyere Blockage"]
        )
        if disturbance == "Drop in Air Flow":
            input_params["air_flow"] = float(input_params.get("air_flow", 1000.0)) * 0.6
            st.caption("Air flow reduced to 60%.")
        elif disturbance == "Spike in Pressure":
            input_params["pressure"] = float(input_params.get("pressure", 3.0)) * 1.6
            st.caption("Pressure increased by 60%.")
        elif disturbance == "Tapping Delay":
            input_params["hot_metal_level"] = float(input_params.get("hot_metal_level", 5.0)) * 1.5
            st.caption("Hot metal accumulates due to delayed tapping.")
        elif disturbance == "Tuyere Blockage":
            input_params["air_flow"] = float(input_params.get("air_flow", 1000.0)) * 0.75
            input_params["temperature"] = float(input_params.get("temperature", 1400.0)) - 100
            st.caption("Reduced air flow and temperature due to blockage.")

    elif scenario == "Slag and Hot Metal Chemistry":
        st.subheader("Slag Basicity & MgO/Al2O3 Impact")
        input_params["slag_rate"] = st.slider("Slag Rate (TPH)", 2.0, 15.0, float(input_params.get("slag_rate", 5.0)), 0.1)
        input_params["CO_content"] = st.slider("MgO/Al2O3 (%)", 10.0, 30.0, float(input_params.get("CO_content", 20.0)), 0.1)
        st.caption("Change slag chemistry for fluidity and lining impact.")

    elif scenario == "Environmental and Energy Compliance":
        st.subheader("Emissions / Energy Cost")
        input_params["pressure"] = st.slider("Blast Pressure (Bar)", 1.0, 6.0, float(input_params.get("pressure", 3.0)), 0.1)
        input_params["feed_rate"] = st.slider("Alternative Fuel Rate (TPH)", 20.0, 90.0, float(input_params.get("feed_rate", 60.0)), 1.0)
        st.caption("Reduce blast or increase alternative fuel to meet new limits.")

    elif scenario == "Production Planning":
        st.subheader("Change Production Target")
        input_params["feed_rate"] = st.slider("Feed Rate (TPH)", 30.0, 120.0, float(input_params.get("feed_rate", 60.0)), 1.0)
        st.caption("Adjust feed rate to match new production demand.")

    elif scenario == "Equipment Failure or Maintenance":
        st.subheader("Stove/Blower Maintenance Simulation")
        equipment_down = st.radio(
            "Equipment Down",
            ["None", "One Stove", "One Blower"]
        )
        if equipment_down == "One Stove":
            input_params["temperature"] = float(input_params.get("temperature", 1400.0)) - 120
            st.caption("Lower blast temperature due to stove down.")
        elif equipment_down == "One Blower":
            input_params["air_flow"] = float(input_params.get("air_flow", 1000.0)) * 0.7
            st.caption("Lower air flow due to blower down.")
        else:
            st.caption("Normal operation.")

    # Run the digital twin model
    st.markdown("#### What-If Results")
    pred = model.simulate_step(input_params)
    st.metric("Predicted Hot Metal Output (TPH)", f"{pred['predicted_hot_metal']:.2f}")

    # Anomaly detection (optional)
    from blast_furnace_model import detect_anomaly
    anomaly = detect_anomaly(input_params)
    if anomaly:
        st.error(f"Anomalies detected: {anomaly}")
    else:
        st.success("No anomalies detected for this scenario.")
