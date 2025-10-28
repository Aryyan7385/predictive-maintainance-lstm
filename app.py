import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

# --- Load Your Saved Assets ---
# Make sure these files are in the same folder
try:
    model = load_model('engine_failure_model.h5')
    scaler = joblib.load('sensor_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler. Make sure 'engine_failure_model.h5' and 'sensor_scaler.pkl' are in this folder. Error: {e}")
    st.stop()

# --- App Configuration ---
st.set_page_config(page_title="Engine Failure Prediction", layout="wide")
st.title("✈️ Predictive Maintenance for Jet Engines")

# --- Helper Function & Constants ---
# This list MUST match the columns you trained on.
# If your training output for 'sensor_cols' was different, change this list!
# This list matches the 19 sensors your model was trained on
SENSOR_COLS = ['sensor' + str(i) for i in range(1, 20)]
SEQUENCE_LENGTH = 50

# --- App UI ---
st.info("Upload a CSV file containing engine sensor data. The model requires at least 50 cycles (rows) of data to make a prediction.")

uploaded_file = st.file_uploader(
    "Upload sensor history (CSV file)"
)

if uploaded_file is not None:
    try:
        # Load the uploaded data
        df = pd.read_csv(uploaded_file)
        
        # --- Data Validation ---
        if not all(col in df.columns for col in SENSOR_COLS):
            st.error(f"Error: File must contain all required sensor columns: {SENSOR_COLS}")
        
        elif len(df) < SEQUENCE_LENGTH:
            st.error(f"Error: Need at least {SEQUENCE_LENGTH} cycles of data. File only has {len(df)}.")
        
        else:
            # --- Prepare Data for Prediction ---
            st.subheader("Latest Sensor Data")
            st.dataframe(df.iloc[-SEQUENCE_LENGTH:][SENSOR_COLS].describe())
            
            # 1. Get the last 50 cycles
            data_to_predict = df.iloc[-SEQUENCE_LENGTH:]
            
            # 2. Scale the sensor data
            scaled_data = scaler.transform(data_to_predict[SENSOR_COLS])
            
            # 3. Reshape for LSTM: (1 sample, 50 time-steps, num_features)
            sequence = np.array([scaled_data])

            # --- Make Prediction ---
            prediction_prob = model.predict(sequence)[0][0]
            prediction_class = (prediction_prob > 0.5).astype(int)

            # --- Display Results ---
            st.subheader(f"Prediction (based on last {SEQUENCE_LENGTH} cycles):")
            
            if prediction_class == 1:
                st.error(
                    f"**WARNING: FAILURE IMMINENT** (Predicted to fail within 30 cycles)"
                )
            else:
                st.success(
                    f"**STATUS: HEALTHY** (Failure not predicted in next 30 cycles)"
                )
                
            st.metric(
                label="Model's Failure Probability", 
                value=f"{prediction_prob * 100:.2f}%"
            )
            
            st.subheader("Sensor Trend (Last 50 Cycles):")
            st.line_chart(data_to_predict[SENSOR_COLS])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")