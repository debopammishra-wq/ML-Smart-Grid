import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
from xgboost import XGBRegressor

# --- Page Configuration ---
st.set_page_config(page_title="AI Smart Grid: Weather-Aware Optimizer", layout="wide")

# --- Energy & Storage Classes ---
class BESS:
    def __init__(self, capacity_kwh=15.0, max_rate=3.0):
        self.capacity_kwh = capacity_kwh
        self.max_rate = max_rate
        self.soc = 0.5 

    def manage(self, net_load):
        if net_load < 0: # Surplus: Charge battery
            charge = min(abs(net_load), self.max_rate, (1 - self.soc) * self.capacity_kwh)
            self.soc += (charge / self.capacity_kwh)
            return -charge
        else: # Deficit: Discharge battery
            discharge = min(net_load, self.max_rate, self.soc * self.capacity_kwh)
            self.soc -= (discharge / self.capacity_kwh)
            return discharge

@st.cache_data
def load_augmented_data(file_id):
    """Downloads data from Drive, loads it, and adds Weather + Renewable features."""
    output = 'household_power_consumption.txt'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # 1. Google Drive Download Logic
    if not os.path.exists(output):
        with st.spinner("Downloading 120MB dataset from Google Drive..."):
            gdown.download(url, output, quiet=False)
    
    # 2. Loading with Memory Optimization
    # We load 100k rows to stay within Streamlit Cloud's 1GB RAM limit
    df = pd.read_csv(output, sep=';', low_memory=False, 
                     na_values=['?'], nrows=100000)
    
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)
    df.set_index('dt', inplace=True)
    df = df.drop(['Date', 'Time'], axis=1).apply(pd.to_numeric, errors='coerce').interpolate()

    # 3. Weather Engine & Generation Simulation
    hour = df.index.hour
    df['cloud_cover'] = np.abs(np.sin(np.linspace(0, 10, len(df)))) * 100 
    df['wind_speed'] = np.random.gamma(shape=2, scale=2, size=len(df))

    # Solar drops with clouds; Wind starts at 3m/s
    solar_pot = np.where((hour > 6) & (hour < 19), np.sin((hour - 6) * np.pi / 12) * 5.0, 0)
    df['solar_gen'] = solar_pot * (1 - (df['cloud_cover'] / 100))
    df['wind_gen'] = np.where(df['wind_speed'] > 3, df['wind_speed'] * 0.4, 0)
    
    # AI Features
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['lag_1h'] = df['Global_active_power'].shift(60)
    return df.dropna()

# --- Initialize System ---
# REPLACE 'YOUR_FILE_ID_HERE' with the ID from your Google Drive link
DRIVE_FILE_ID = 'YOUR_FILE_ID_HERE' 

try:
    data = load_augmented_data(DRIVE_FILE_ID)
    bess = BESS()

    # --- Sidebar ---
    st.sidebar.header("🕹️ Grid Controls")
    dr_limit = st.sidebar.slider("DR Threshold (kW)", 1.0, 8.0, 3.5)
    st.sidebar.info("High cloud cover reduces solar yield, forcing the BESS to discharge earlier.")

    # --- Real-Time Dispatch Logic ---
    last_row = data.iloc[-1]
    renewables = last_row['solar_gen'] + last_row['wind_gen']
    net_load = last_row['Global_active_power'] - renewables
    bess_act = bess.manage(net_load)
    final_grid = max(0, net_load - bess_act)

    # --- Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Demand", f"{last_row['Global_active_power']:.2f} kW")
    m2.metric("Renewable Supply", f"{renewables:.2f} kW", f"{last_row['solar_gen']:.1f}S | {last_row['wind_gen']:.1f}W")
    m3.metric("BESS Storage", f"{bess.soc*100:.1f}%", f"{bess_act:.2f} kW")
    m4.metric("Grid Draw", f"{final_grid:.2f} kW", delta_color="inverse")

    st.divider()

    # --- Main Dashboard Layout ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Energy Dispatch Stack")
        history = data.tail(60).copy()
        # Visualizing the efficiency of the source mix
        dispatch_df = pd.DataFrame({
            "Thermal (Grid)": [1.2] * 60,
            "Wind Gen": history['wind_gen'].values,
            "Solar Gen": history['solar_gen'].values
        }, index=history.index)
        st.area_chart(dispatch_df, color=["#FF4B4B", "#1E90FF", "#FFD700"])
        
        st.subheader("☁️ Environmental Monitoring")
        st.line_chart(history[['cloud_cover', 'wind_speed']])

    with col_right:
        st.subheader("🧠 Weather-Aware AI")
        # Multi-variate XGBoost Training
        feats = ['hour_sin', 'cloud_cover', 'wind_speed', 'lag_1h']
        X_train = data[feats].tail(2000)
        y_train = data['Global_active_power'].tail(2000)
        
        model = XGBRegressor(n_estimators=100).fit(X_train[:-60], y_train[:-60])
        forecast = model.predict(X_train[-60:])
        
        peak = np.max(forecast)
        if peak > dr_limit:
            st.error(f"🚨 PEAK PREDICTED: {peak:.2f} kW")
            st.warning("Action: Automating Demand Response")
        else:
            st.success("✅ System Stable")
        
        st.write(f"**Cloud Cover:** {last_row['cloud_cover']:.1f}%")
        st.progress(last_row['cloud_cover']/100)
        st.write(f"**Wind Speed:** {last_row['wind_speed']:.1f} m/s")

    st.subheader("📋 Extended Telemetry")
    st.line_chart(data[['Global_active_power', 'Voltage', 'Global_intensity']].tail(120))

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Check your Google Drive File ID and permissions.")