
# NEO Detection ML Project

A machine learning pipeline and dashboard for Near-Earth Object (NEO) detection simulation and analysis.

## Features

- **Physics Simulation**: Simulates NEO orbits and calculating apparent magnitudes using SPICE kernels and H-G photometry.
- **Data Generation**: Generates synthetic observation datasets for training.
- **Machine Learning**: 
  - `RandomForestRegressor` for apparent magnitude prediction.
  - `RandomForestClassifier` for detection probability (visible vs not).
- **Interactive Dashboard**: Streamlit app for real-time "What-If" analysis of detection capabilities.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Requires `spiceypy`, `pandas`, `numpy`, `scikit-learn`, `streamlit`, `joblib`, `plotly`, `matplotlib`, `seaborn`)*

## Usage

### 1. Setup Data
Data is simulated based on a mock database for demonstration.
```bash
python scripts/create_mock_db.py
python scripts/generate_data.py --neos 500 --start 2026-01-01 --end 2026-06-01 --step-hours 24 --limiting-mag 22.0 --exposure-s 30 --fov-deg 90.0
```

### 2. Train Models
```bash
python src/ml/train_regressor.py
python src/ml/train_classifier.py
```

### 3. Run Dashboard
```bash
streamlit run dashboard/app.py
```

## Structure

- `src/`: Core logic (physics, simulation, ml).
- `scripts/`: CLI tools for data management.
- `dashboard/`: Streamlit application.
- `data/`: Storage for simulated observations.
- `models/`: Trained model artifacts.
