
import streamlit as st
import pandas as pd
import joblib
import sys
import os
import pathlib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define features directly to avoid import issues
FEATURE_COLUMNS = [
    "a", "e", "i", "H", 
    "distance_au", "phase_angle_deg", 
    "ra_deg", "dec_deg", 
    "limiting_mag", "exposure_s", "fov_deg"
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
    limiting_mag = st.sidebar.slider("Limiting Magnitude", 15.0, 25.0, 22.0, 0.1)
    exposure_s = st.sidebar.slider("Exposure Time (s)", 10.0, 300.0, 30.0, 10.0)
    fov_deg = st.sidebar.slider("Field of View (deg)", 0.5, 10.0, 2.0, 0.1)
    
    # Filter Data for Date
    day_data = df[df["date"] == selected_date].copy()
    st.write(f"**{len(day_data)}** objects tracked on {selected_date}")

    # Inject Parameters (What-If)
    day_data["limiting_mag"] = limiting_mag
    day_data["exposure_s"] = exposure_s
    day_data["fov_deg"] = fov_deg

    # Load Models & Predict
    reg, clf = load_models()
    
    if reg and clf:
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

if __name__ == "__main__":
    main()
