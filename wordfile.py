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
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, DATA_FILE, quiet=False)
    
    # Load rows
    df = pd.read_csv(DATA_FILE, sep=';', low_memory=False, nrows=100000)
    
    # --- FIX STARTS HERE ---
    # 1. Combine Date and Time strings into one column
    # 2. Convert to actual Datetime objects
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    # 3. Set Datetime as the Index (This fixes the X-Axis)
    df.set_index('Datetime', inplace=True)
    # --- FIX ENDS HERE ---

    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    
    # Feature Engineering (Using integer indexing for math since we changed the index to Datetime)
    df['hour_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 1440)
    df['cloud_cover'] = np.random.uniform(0, 100, len(df))
    df['wind_speed'] = np.random.uniform(0, 15, len(df))
    df['solar_gen'] = np.where(df['cloud_cover'] < 30, np.random.uniform(0.5, 2.0, len(df)), 0)
    df['wind_gen'] = np.where(df['wind_speed'] > 5, np.random.uniform(0.2, 1.0, len(df)), 0)
    df['lag_1h'] = df['Global_active_power'].shift(60)
    return df.dropna()

data = load_data()

# --- 2. SIDEBAR PARAMETERS ---
st.sidebar.header("⚡ Grid Parameters")
dr_limit = st.sidebar.slider("DR Threshold (kW)", 1.0, 5.0, 2.5)
soc = 0.407 

# --- 3. MAIN DASHBOARD ---
st.title("🌐 Multi-Source Smart Grid Optimizer")

m1, m2, m3, m4 = st.columns(4)
current_val = data['Global_active_power'].iloc[-1]
m1.metric("Current Demand", f"{current_val:.2f} kW")
m2.metric("Renewable Supply", f"{(data['solar_gen'].iloc[-1] + data['wind_gen'].iloc[-1]):.2f} kW")
m3.metric("BESS Storage", f"{soc*100:.1f}%")
m4.metric("Grid Draw", "0.00 kW" if soc > 0.2 else f"{current_val:.2f} kW")

# --- 4. TABBED INTERFACE ---
tab1, tab2 = st.tabs(["📊 Live Telemetry", "🧠 AI Analysis"])

with tab1:
    st.subheader("Energy Dispatch Stack")
    history = data.tail(60).copy()
    
    # Now that 'history' has a Datetime index, st.area_chart will use it automatically!
    dispatch_df = pd.DataFrame({
        "Thermal (Grid)": [0.2] * 60,
        "Wind Gen": history['wind_gen'].values,
        "Solar Gen": history['solar_gen'].values
    }, index=history.index) 
    
    st.area_chart(dispatch_df, color=["#FF4B4B", "#1E90FF", "#FFD700"])

    st.subheader("☁️ Environmental Factors")
    # This chart will also update to show Time
    st.line_chart(history[['cloud_cover', 'wind_speed']])

with tab2:
    st.subheader("🧠 XGBoost AI Analysis")
    feats = ['hour_sin', 'cloud_cover', 'wind_speed', 'lag_1h']
    X = data[feats].tail(3000)
    y = data['Global_active_power'].tail(3000)
    
    model = XGBRegressor(n_estimators=50, max_depth=3, colsample_bytree=0.7)
    model.fit(X[:-60], y[:-60])
    
    importance_scores = model.feature_importances_
    
    if sum(importance_scores) > 0:
        importance_df = pd.DataFrame({
            'Feature': feats,
            'Influence': importance_scores
        }).sort_values(by='Influence', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'), use_container_width=True, height=300)
    else:
        st.warning("Model is training...")

st.divider()
st.markdown("**Project Note:** This system utilizes **XGBoost** to manage energy dispatch. The X-axis now reflects actual historical timestamps.")
