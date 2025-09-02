import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set page configuration to wide layout
st.set_page_config(
    page_title="Network IDS",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': 'A Deep Learning Anomaly Detection Project.'
    }
)

# Inject CSS for Font Awesome icons and footer styling
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            color: #8B93A7;
            background-color: #0e1117;
        }
        .social-icons a {
            color: #8B93A7;
            font-size: 20px;
            margin: 0 10px;
            text-decoration: none;
            transition: color 0.3s;
        }
        .social-icons a:hover {
            color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. Load the saved model, threshold, and scaler (Cached to run once)
# ==============================================================================
@st.cache_resource
def load_model_components():
    """Loads the model, scaler, and threshold files."""
    try:
        autoencoder = load_model(
            'autoencoder_model.h5',
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        with open('threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return autoencoder, threshold, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure the files are in the same directory.")
        return None, None, None

# Load the components at the start of the app
autoencoder, threshold, scaler = load_model_components()

# ==============================================================================
# 2. Function to preprocess and predict
# ==============================================================================
def preprocess_and_predict(df, scaler, autoencoder, threshold):
    """Preprocesses the DataFrame and returns predictions."""
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Flow ID', 'Unnamed: 0'], errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    required_cols = scaler.feature_names_in_
    df_processed = df.reindex(columns=required_cols, fill_value=0)
    X_scaled = pd.DataFrame(scaler.transform(df_processed), columns=df_processed.columns)
    
    reconstructed_data = autoencoder.predict(X_scaled.values)
    mse = np.mean(np.power(X_scaled.values - reconstructed_data, 2), axis=1)
    predictions = (mse > threshold).astype(int)
    
    return predictions, mse

# ==============================================================================
# 3. Streamlit UI and Logic
# ==============================================================================
# --- Header Section ---
st.title("Network Anomaly Detection System")
st.markdown("### Powered by a Deep Learning Autoencoder")
st.markdown("A prototype for identifying malicious network traffic by detecting deviations from normal behavior.")
st.markdown("---")

# --- Main Content Section ---
with st.container(border=True):
    st.subheader("Upload Your Network Traffic Data")
    uploaded_file = st.file_uploader("Choose a CSV file (from CICFlowMeter)", type="csv")

    if uploaded_file is not None and autoencoder is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.info("File uploaded successfully. Starting analysis...")
            
            # Use a spinner for a sleek loading animation
            with st.spinner("Analyzing network traffic... This may take a moment."):
                predictions, mse = preprocess_and_predict(df.copy(), scaler, autoencoder, threshold)
            
            # --- Results Section ---
            st.success("Analysis Complete!")
            st.markdown("---")
            st.subheader("Prediction Results")
            
            df['Reconstruction Error'] = mse
            df['Prediction'] = np.where(predictions == 1, 'Anomaly (Attack)', 'Normal (Benign)')

            anomalies = df[df['Prediction'] == 'Anomaly (Attack)']
            
            if not anomalies.empty:
                st.error(f"⚠️ **Found {len(anomalies)} flows flagged as ANOMALY:**")
                st.dataframe(anomalies, use_container_width=True)
            else:
                st.info("✅ **No anomalies were detected in the file.**")
                st.dataframe(df.head(20), use_container_width=True)
                st.markdown("*(Showing the first 20 flows)*")
        
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# ==============================================================================
# 4. Footer Section
# ==============================================================================
st.markdown("---")
# Custom HTML for the footer
footer_html = """
<div style="text-align: center; color: #8B93A7; font-size: 14px; margin-bottom: 5px;">
    Project by Dipankar Saha
</div>
<div class="social-icons" style="text-align: center;">
    <a href="mailto:dsaha0427@gmail.com" title="Mail">
        <i class="fas fa-envelope"></i>
    </a>
    <a href="https://www.linkedin.com/in/dipankarsaha2001/" target="_blank" title="LinkedIn">
        <i class="fab fa-linkedin"></i>
    </a>
    <a href="https://github.com/ezFaith" target="_blank" title="GitHub">
        <i class="fab fa-github"></i>
    </a>
    <a href="https://ezfaith.github.io/portfolio/" target="_blank" title="Portfolio">
        <i class="fas fa-briefcase"></i>
    </a>
</div>
<div style="text-align: center; color: #8B93A7; font-size: 12px; margin-top: 10px;">
    A cybersecurity project using a Deep Learning Autoencoder.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)