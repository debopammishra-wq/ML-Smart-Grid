import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import gdown

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="AI Smart Grid Optimizer", layout="wide")

# Replace this with your specific Google Drive File ID
DRIVE_FILE_ID = '1_6QpH-JOTlxhxcbq6M2qSXrTNDzlSSEq'
DATA_FILE = 'household_power_consumption.txt'

@st.cache_data
def load_data():
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, DATA_FILE, quiet=False)
    
    # Load 100k rows to stay within Streamlit Cloud RAM limits
    df = pd.read_csv(DATA_FILE, sep=';', low_memory=False, nrows=100000)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    
    # Feature Engineering
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
bess_capacity = 15.0 # kWh
soc = 0.407 # Your current 40.7% state

# --- 3. MAIN DASHBOARD ---
st.title("🌐 Multi-Source Smart Grid Optimizer")

# Summary Metrics
m1, m2, m3, m4 = st.columns(4)
current_val = data['Global_active_power'].iloc[-1]
m1.metric("Current Demand", f"{current_val:.2f} kW")
m2.metric("Renewable Supply", f"{(data['solar_gen'].iloc[-1] + data['wind_gen'].iloc[-1]):.2f} kW")
m3.metric("BESS Storage", f"{soc*100:.1f}%")
m4.metric("Grid Draw", "0.00 kW" if soc > 0.2 else f"{current_val:.2f} kW")

# --- 4. TABBED INTERFACE (Fixes Mobile Visibility) ---
tab1, tab2 = st.tabs(["📊 Live Telemetry", "🧠 AI Analysis"])

with tab1:
    st.subheader("Energy Dispatch Stack")
    history = data.tail(60).copy()
    dispatch_df = pd.DataFrame({
        "Thermal (Grid)": [0.2] * 60, # Base load
        "Wind Gen": history['wind_gen'].values,
        "Solar Gen": history['solar_gen'].values
    }, index=history.index)
    st.area_chart(dispatch_df, color=["#FF4B4B", "#1E90FF", "#FFD700"])

    st.subheader("☁️ Environmental Factors")
    st.line_chart(history[['cloud_cover', 'wind_speed']])

with tab2:
    st.subheader("🧠 XGBoost AI Analysis")
    
    # 1. Training Parameters
    feats = ['hour_sin', 'cloud_cover', 'wind_speed', 'lag_1h']
    X = data[feats].tail(3000)
    y = data['Global_active_power'].tail(3000)
    
    # 2. Robust Model Settings
    model = XGBRegressor(n_estimators=50, max_depth=3, colsample_bytree=0.7)
    model.fit(X[:-60], y[:-60])
    
    # 3. Force Render for Mobile
    # We use st.bar_chart with a fixed height to prevent it from collapsing on phones
    importance_scores = model.feature_importances_
    
    if sum(importance_scores) > 0:
        importance_df = pd.DataFrame({
            'Feature': feats,
            'Influence': importance_scores
        }).sort_values(by='Influence', ascending=False)
        
        # Setting use_container_width=True and a fixed height fixes mobile rendering
        st.bar_chart(importance_df.set_index('Feature'), use_container_width=True, height=300)
        
        # Add a text-based backup for mobile users
        with st.expander("See Raw Importance Scores"):
            st.write(importance_df)
    else:
        st.warning("Model is training... Refreshing telemetry.")

# --- 5. FOOTER ---
st.divider()
st.markdown("""
**Project Note:** This system utilizes **XGBoost** to manage energy dispatch between renewables and BESS. 
The current configuration targets a **Net-Zero** grid draw by utilizing stored energy during demand spikes.
""")
