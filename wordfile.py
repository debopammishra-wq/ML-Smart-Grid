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
    
    # Combined Datetime Fix
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('Datetime', inplace=True)
    
    # Clean Power Data
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    
    # --- THE FIX: FEATURE ENGINEERING ---
    # We use a 5-minute lag for high-accuracy near-real-time tracking
    df['lag_5m'] = df['Global_active_power'].shift(5)
    df['hour_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 1440)
    
    # Synthetic Weather
    df['cloud_cover'] = np.random.uniform(0, 100, len(df))
    df['wind_speed'] = np.random.uniform(0, 15, len(df))
    
    # Renewable Logic
    df['solar_gen'] = np.where(df['cloud_cover'] < 30, np.random.uniform(0.5, 2.0, len(df)), 0)
    df['wind_gen'] = np.where(df['wind_speed'] > 5, np.random.uniform(0.2, 1.0, len(df)), 0)
    
    return df.dropna()

# Initialize Data
data = load_data()

# --- 2. THE FIX: HIGH-PERFORMANCE AI TRAINING ---
# We train before the UI to ensure the forecast line is ready
feats = ['hour_sin', 'cloud_cover', 'wind_speed', 'lag_5m']
X = data[feats].tail(10000) 
y = data['Global_active_power'].tail(10000)

# Optimized XGBoost parameters for tight curve fitting
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
model.fit(X, y)

# --- 3. SIDEBAR ---
st.sidebar.header("⚡ Grid Parameters")
dr_limit = st.sidebar.slider("Demand Response Threshold (kW)", 1.0, 5.0, 2.5)
soc = 0.407 # State of Charge

# --- 4. MAIN DASHBOARD ---
st.title("🌐 Multi-Source Smart Grid Optimizer")

m1, m2, m3, m4 = st.columns(4)
current_val = data['Global_active_power'].iloc[-1]
current_renewables = data['solar_gen'].iloc[-1] + data['wind_gen'].iloc[-1]

m1.metric("Current Demand", f"{current_val:.2f} kW")
m2.metric("Renewable Supply", f"{current_renewables:.2f} kW")
m3.metric("BESS Storage", f"{soc*100:.1f}%")
m4.metric("Grid Status", "Stable" if current_val < dr_limit else "Peak Alert")

# --- 5. TABS ---
tab1, tab2 = st.tabs(["📊 Live Telemetry & Forecast", "🧠 AI Model Insights"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚡ High-Accuracy Load Forecast")
        history = data.tail(60).copy()
        
        # Inference
        forecast_input = history[feats]
        predictions = model.predict(forecast_input)
        
        # Comparison DataFrame
        forecast_df = pd.DataFrame({
            "Actual Demand": history['Global_active_power'].values,
            "AI Forecast": predictions
        }, index=history.index)
        
        # Line chart for the AI prediction
        st.line_chart(forecast_df, color=["#2ecc71", "#e74c3c"])
        
        st.subheader("🔋 Energy Dispatch Stack")
        dispatch_df = pd.DataFrame({
            "Grid (Thermal)": [0.5] * 60,
            "Wind": history['wind_gen'].values,
            "Solar": history['solar_gen'].values
        }, index=history.index)
        st.area_chart(dispatch_df, color=["#34495e", "#3498db", "#f1c40f"])

    with col2:
        st.subheader("☁️ Environment")
        st.line_chart(history[['cloud_cover', 'wind_speed']])
        
        if current_val > dr_limit:
            st.warning(f"⚠️ High Demand Alert: {current_val:.2f}kW")
        else:
            st.success("✅ Load levels optimal.")

with tab2:
    st.subheader("🧠 XGBoost Feature Influence")
    importance_df = pd.DataFrame({
        'Feature': feats,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'), color="#9b59b6")

# --- 6. FOOTER ---
st.divider()
st.markdown("**Note:** Forecast accuracy optimized using a 5-minute Autoregressive Lag feature.")
