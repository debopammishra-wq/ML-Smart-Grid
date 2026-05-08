import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import gdown

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="AI Smart Grid Optimizer", layout="wide")

DRIVE_FILE_ID = '1_6QpH-JOTlxhxcbq6M2qSXrTNDzlSSEq'
DATA_FILE = 'household_power_consumption.txt'

@st.cache_data
def load_data():
    # Download from Google Drive
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, DATA_FILE, quiet=False)
    
    # Load data with Datetime Indexing
    df = pd.read_csv(DATA_FILE, sep=';', low_memory=False, nrows=80000)
    
    # Combine Date and Time for the X-Axis fix
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('Datetime', inplace=True)
    
    # Convert power to numeric
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    
    # Feature Engineering (Weather is synthetic, Power is real)
    df['hour_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 1440)
    df['cloud_cover'] = np.random.uniform(0, 100, len(df))
    df['wind_speed'] = np.random.uniform(0, 15, len(df))
    
    # Logic for Renewable Generation
    df['solar_gen'] = np.where(df['cloud_cover'] < 30, np.random.uniform(0.5, 2.0, len(df)), 0)
    df['wind_gen'] = np.where(df['wind_speed'] > 5, np.random.uniform(0.2, 1.0, len(df)), 0)
    
    # Lag feature for the AI
    df['lag_1h'] = df['Global_active_power'].shift(60)
    return df.dropna()

# Initialize Data
data = load_data()

# --- 2. AI MODEL TRAINING (Pre-calculating for Tabs) ---
feats = ['hour_sin', 'cloud_cover', 'wind_speed', 'lag_1h']
X = data[feats].tail(5000)  # Use last 5000 rows for training
y = data['Global_active_power'].tail(5000)

# Train XGBoost
model = XGBRegressor(n_estimators=100, max_depth=4, eta=0.1)
model.fit(X[:-60], y[:-60]) # Train on all but the last hour

# --- 3. SIDEBAR PARAMETERS ---
st.sidebar.header("⚡ Grid Parameters")
dr_limit = st.sidebar.slider("Demand Response Threshold (kW)", 1.0, 5.0, 2.5)
soc = 0.407 # Battery State of Charge

# --- 4. MAIN DASHBOARD ---
st.title("🌐 Multi-Source Smart Grid Optimizer")

# Real-time Metrics
m1, m2, m3, m4 = st.columns(4)
current_val = data['Global_active_power'].iloc[-1]
current_renewables = data['solar_gen'].iloc[-1] + data['wind_gen'].iloc[-1]

m1.metric("Current Demand", f"{current_val:.2f} kW")
m2.metric("Renewable Supply", f"{current_renewables:.2f} kW")
m3.metric("BESS Storage (SOC)", f"{soc*100:.1f}%")
m4.metric("Grid Status", "Stabilized" if current_val < dr_limit else "Peak Alert", delta_color="inverse")

# --- 5. TABBED INTERFACE ---
tab1, tab2 = st.tabs(["📊 Live Telemetry & Forecast", "🧠 AI Model Insights"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚡ Energy Dispatch & Load Forecast")
        history = data.tail(60).copy()
        
        # 1. Prediction Logic
        forecast_input = history[feats]
        predictions = model.predict(forecast_input)
        
        # 2. Plotting Actual vs Forecast
        forecast_df = pd.DataFrame({
            "Actual Demand": history['Global_active_power'].values,
            "AI Forecast": predictions
        }, index=history.index)
        st.line_chart(forecast_df, color=["#2ecc71", "#e74c3c"])
        
        # 3. Generation Mix Stack
        st.subheader("🔋 Generation Stack (Thermal vs. Renewables)")
        dispatch_df = pd.DataFrame({
            "Grid (Thermal)": [0.5] * 60,
            "Wind": history['wind_gen'].values,
            "Solar": history['solar_gen'].values
        }, index=history.index)
        st.area_chart(dispatch_df, color=["#34495e", "#3498db", "#f1c40f"])

    with col2:
        st.subheader("☁️ Environment")
        st.write("Real-time weather impact on DERs")
        st.line_chart(history[['cloud_cover', 'wind_speed']])
        
        if current_val > dr_limit:
            st.warning(f"⚠️ Demand ({current_val:.2f}kW) exceeds threshold! Triggering BESS Discharge.")
        else:
            st.success("✅ Grid operating within safe limits.")

with tab2:
    st.subheader("🧠 XGBoost Feature Importance")
    st.write("Which factors are driving the current load predictions?")
    
    importance_df = pd.DataFrame({
        'Feature': feats,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'), color="#9b59b6")
    
    with st.expander("Model Technical Details"):
        st.write(f"Training Samples: {len(X)}")
        st.write("Algorithm: Extreme Gradient Boosting (XGBoost)")
        st.write("Objective: reg:squarederror")

# --- 6. FOOTER ---
st.divider()
st.markdown("""
**Data Provenance:** Electrical data sourced from UCI Machine Learning Repository. Weather data (Wind/Cloud) is synthetically generated to demonstrate multivariate XGBoost forecasting.
""")
