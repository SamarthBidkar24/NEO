
import streamlit as st
import pandas as pd
import joblib
import sys
import os
import pathlib
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define features directly to avoid import issues
FEATURE_COLUMNS = [
    "a", "e", "i", "H", 
    "distance_au", "phase_angle_deg", 
    "ra_deg", "dec_deg", 
    "limiting_mag", "exposure_s", "fov_deg",
    "eff_limiting_mag", "in_frame"
]

# Page Config
st.set_page_config(page_title="NEO Dashboard", layout="wide")

# Paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "simulated_observations.csv"
MODELS_DIR = PROJECT_ROOT / "models"

def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data missing at {DATA_PATH}")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_PATH)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        df["date"] = df["timestamp_utc"].dt.date
    return df

def load_models():
    try:
        reg = joblib.load(MODELS_DIR / "regressor.joblib")
        clf = joblib.load(MODELS_DIR / "classifier.joblib")
        return reg, clf
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

def main():
    st.title("NEO Detection Analysis")

    # Load Data
    df = load_data()
    if df.empty:
        return

    # Sidebar
    st.sidebar.header("1. Select Observation Date")
    unique_dates = sorted(df["date"].unique())
    selected_date = st.sidebar.selectbox("Date", unique_dates)

    st.sidebar.header("2. Camera Settings (Simulation)")
    base_limiting_mag = st.sidebar.slider("Base Limiting Magnitude (at 30s)", 15.0, 25.0, 22.0, 0.1)
    exposure_s = st.sidebar.slider("Exposure Time (s)", 10.0, 300.0, 30.0, 10.0)
    
    # Calculate Effective Limiting Magnitude based on Exposure
    # Mag gains ~1.25 per 10x exposure increase (sqrt relationship)
    limiting_mag = base_limiting_mag + 1.25 * np.log10(exposure_s / 30.0)
    st.sidebar.caption(f"Effective Limiting Mag: **{limiting_mag:.2f}**")
    fov_deg = st.sidebar.slider("Field of View (deg)", 0.5, 10.0, 2.0, 0.1)
    
    # Filter Data based on Regimes (Coarse Filtering)
    df_filtered = df[df["date"] == selected_date].copy()

    # 1. Limiting Magnitude -> Difficulty Regime
    # Bright (<=18), Medium (18-21), Faint (>21)
    if limiting_mag <= 18:
        current_regime = "Bright Objects (Mag <= 18)"
        df_filtered = df_filtered[df_filtered["apparent_magnitude"] <= 18]
    elif limiting_mag <= 21:
        current_regime = "Medium Objects (18 < Mag <= 21)"
        df_filtered = df_filtered[(df_filtered["apparent_magnitude"] > 18) & (df_filtered["apparent_magnitude"] <= 21)]
    else:
        current_regime = "Faint Objects (Mag > 21)"
        df_filtered = df_filtered[df_filtered["apparent_magnitude"] > 21]

    # 2. Exposure Bucket
    # Short (<60s) vs Long (>=60s)
    if "exposure_s" in df_filtered.columns:
        if exposure_s < 60:
            df_filtered = df_filtered[df_filtered["exposure_s"] <= 60]
        else:
            df_filtered = df_filtered[df_filtered["exposure_s"] > 60]

    # 3. FOV Bucket
    # Narrow (<2.0 deg) vs Wide (>=2.0 deg)
    if "fov_deg" in df_filtered.columns:
        if fov_deg < 2.0:
            df_filtered = df_filtered[df_filtered["fov_deg"] <= 2.0]
        else:
            df_filtered = df_filtered[df_filtered["fov_deg"] > 2.0]

    # Sampling
    max_points = 500
    if len(df_filtered) > max_points:
        df_filtered = df_filtered.sample(max_points, random_state=42)
    
    # Display Summary
    st.markdown(f"**Showing {len(df_filtered)} samples** | Regime: {current_regime} | Exposure: {exposure_s}s | FOV: {fov_deg}°")

    # Inject Parameters (What-If) into the filtered dataset
    # This overwrites the actual values with the slider values for simulation/prediction
    # Inject Parameters (What-If) into the filtered dataset
    # We re-compute features based on sliders to simulate "What-If"
    # 1. Effective Limiting Magnitude (Sensitivity)
    # 'limiting_mag' variable from sidebar holds the calculated Effective Mag
    df_filtered["eff_limiting_mag"] = limiting_mag 
    df_filtered["limiting_mag"] = base_limiting_mag # Dataset expects 'limiting_mag' as Base
    
    # 2. Exposure & FOV
    df_filtered["exposure_s"] = exposure_s
    df_filtered["fov_deg"] = fov_deg
    
    # 3. In-Frame Check (Pointing)
    # We simulate a "Pointing" constraint.
    # If the user sets FOV=2.0, we only detect objects within 1.0 deg of the opposition center.
    # 'off_axis_angle_deg' must exist in the dataset (computed relative to opposition).
    if "off_axis_angle_deg" in df_filtered.columns:
        # Calculate in_frame dynamically
        df_filtered["in_frame"] = (df_filtered["off_axis_angle_deg"] <= (fov_deg / 2.0)).astype(int)
    else:
        # Fallback if column missing (e.g. old data)
        df_filtered["in_frame"] = 1
        
    day_data = df_filtered
    
    # DEBUG: Show what's happening
    st.divider()
    dbg_col1, dbg_col2, dbg_col3 = st.columns(3)
    dbg_col1.metric("Model Input Mag Limit", f"{limiting_mag:.2f}")
    dbg_col2.metric("Model Input Exposure", f"{exposure_s:.1f}s")
    
    # Load Models & Predict
    reg, clf = load_models()
    
    if reg and clf:
        if not day_data.empty:
            X = day_data[FEATURE_COLUMNS]
            day_data["apparent_mag_pred"] = reg.predict(X)
            day_data["detected_prob"] = clf.predict_proba(X)[:, 1]
            
            # Plotting
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sky Map (Detection Probability)")
                st.scatter_chart(
                    day_data,
                    x="ra_deg",
                    y="dec_deg",
                    color="detected_prob",
                    size=50 # Explicit size
                )
                
            with col2:
                st.subheader("Magnitude Analysis")
                st.scatter_chart(
                    day_data,
                    x="apparent_magnitude",
                    y="apparent_mag_pred",
                    color="detected_prob"
                )
            
            # Data Table
            with st.expander("View Raw Data"):
                st.dataframe(day_data)
        else:
            st.warning("No objects found in this regime. Try adjusting the filters.")

if __name__ == "__main__":
    main()
