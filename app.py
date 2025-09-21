import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class for better code organization
class Config:
    """Application configuration constants."""
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    BATCH_SIZE = 1000
    MODEL_FILES = {
        'autoencoder': 'autoencoder_model.h5',
        'threshold': 'threshold.pkl',
        'scaler': 'scaler.pkl'
    }
    UNWANTED_COLUMNS = ['Flow ID', 'Unnamed: 0']
    PRESERVE_COLUMNS = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']
    COLORS = {
        'benign': '#2E86AB',
        'anomaly': '#F24236',
        'threshold': '#FF6B35'
    }

# Set page configuration
st.set_page_config(
    page_title="ML-IDS",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'A Machine Learning Based Network Intrusion Detection System.'
    }
)

# Get theme colors with fallbacks
@st.cache_data
def get_theme_colors():
    """Get theme colors with fallbacks."""
    return {
        'primary': st.get_option("theme.primaryColor") or "#FF6B35",
        'background': st.get_option("theme.backgroundColor") or "#FFFFFF",
        'secondary_background': st.get_option("theme.secondaryBackgroundColor") or "#F0F2F6",
        'text': st.get_option("theme.textColor") or "#262730"
    }

theme_colors = get_theme_colors()

# Enhanced CSS styling
st.markdown(f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* Custom Title Banner Styling */
        .title-banner {{
            background: linear-gradient(135deg, {theme_colors['secondary_background']}, {theme_colors['primary']}20);
            padding: 2rem 0;
            margin-bottom: 2rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid {theme_colors['primary']}30;
        }}
        .title-banner h1 {{
            font-size: 2.5em;
            color: {theme_colors['text']};
            margin: 0;
            font-family: 'Poppins', sans-serif;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .title-description {{
            text-align: center;
            font-family: 'Inter', sans-serif;
            color: {theme_colors['text']};
            margin-top: -1rem;
            margin-bottom: 2rem;
        }}
        /* Enhanced button styling */
        .stButton>button {{
            border: 2px solid {theme_colors['primary']};
            color: {theme_colors['primary']};
            background-color: transparent;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            color: white;
            background-color: {theme_colors['primary']};
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }}
        /* Enhanced metrics styling */
        .stMetric {{
            border: 1px solid {theme_colors['secondary_background']};
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, {theme_colors['secondary_background']}, {theme_colors['background']});
        }}
        /* Status indicators */
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-error {{
            color: #dc3545;
            font-weight: bold;
        }}
        .status-warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        /* Social icons styling */
        .social-icons a {{
            color: {theme_colors['text']};
            font-size: 20px;
            margin: 0 10px;
            text-decoration: none;
            transition: all 0.3s ease;
        }}
        .social-icons a:hover {{
            color: {theme_colors['primary']};
            transform: translateY(-2px);
        }}
        /* Font imports */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Inter:wght@400&display=swap');
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Enhanced Model Loading with Better Error Handling
# ==============================================================================
@st.cache_resource
def load_model_components() -> Tuple[Optional[tf.keras.Model], Optional[float], Optional[object]]:
    """Loads the model, scaler, and threshold files with comprehensive error handling."""
    try:
        # Check if all required files exist
        missing_files = []
        for file_type, filename in Config.MODEL_FILES.items():
            if not Path(filename).exists():
                missing_files.append(f"{filename} ({file_type})")
        
        if missing_files:
            st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            st.info("Please ensure all model files are in the same directory as this application.")
            return None, None, None
        
        # Load model with custom objects
        autoencoder = load_model(
            Config.MODEL_FILES['autoencoder'],
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        
        # Load threshold
        with open(Config.MODEL_FILES['threshold'], 'rb') as f:
            threshold = pickle.load(f)
        
        # Load scaler
        with open(Config.MODEL_FILES['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info("All model components loaded successfully")
        return autoencoder, threshold, scaler
        
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        st.error(f"❌ Error loading model components: {e}")
        return None, None, None

# ==============================================================================
# Enhanced Data Processing with Progress Tracking
# ==============================================================================
def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate the uploaded DataFrame."""
    if df.empty:
        st.error("❌ The uploaded file is empty.")
        return False
    
    if len(df.columns) < 5:
        st.warning("⚠️ File seems to have too few columns. Please verify it's a CICFlowMeter output.")
    
    # Check for minimum required numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 3:
        st.error("❌ Not enough numeric columns found. Please check your data format.")
        return False
    
    return True

def preprocess_and_predict(
    df: pd.DataFrame, 
    scaler: object, 
    autoencoder: tf.keras.Model, 
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Enhanced preprocessing with progress tracking and batch processing."""
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data cleaning (25%)
        status_text.text("🧹 Step 1/4: Cleaning and preparing data...")
        progress_bar.progress(25)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove unwanted columns more efficiently
        cols_to_drop = []
        for col in df.columns:
            if col in Config.UNWANTED_COLUMNS:
                cols_to_drop.append(col)
            elif df[col].dtype == 'object' and col not in Config.PRESERVE_COLUMNS:
                cols_to_drop.append(col)
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Handle timestamp if present
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Handle infinite values and NaN more efficiently
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Step 2: Feature alignment (50%)
        status_text.text("🔧 Step 2/4: Aligning features with model requirements...")
        progress_bar.progress(50)
        
        # Ensure we have the required columns for the scaler
        required_cols = scaler.feature_names_in_
        df_processed = df.reindex(columns=required_cols, fill_value=0)
        
        # Step 3: Scaling and prediction (75%)
        status_text.text("⚡ Step 3/4: Scaling features and running predictions...")
        progress_bar.progress(75)
        
        # Scale the features
        X_scaled = pd.DataFrame(
            scaler.transform(df_processed), 
            columns=df_processed.columns,
            index=df_processed.index
        )
        
        # Batch processing for large datasets
        if len(X_scaled) > Config.BATCH_SIZE:
            predictions = []
            mse_values = []
            
            for i in range(0, len(X_scaled), Config.BATCH_SIZE):
                batch = X_scaled.iloc[i:i+Config.BATCH_SIZE]
                reconstructed_batch = autoencoder.predict(batch.values, verbose=0)
                mse_batch = np.mean(np.power(batch.values - reconstructed_batch, 2), axis=1)
                pred_batch = (mse_batch > threshold).astype(int)
                
                predictions.extend(pred_batch)
                mse_values.extend(mse_batch)
            
            predictions = np.array(predictions)
            mse = np.array(mse_values)
        else:
            # Process all at once for smaller datasets
            reconstructed_data = autoencoder.predict(X_scaled.values, verbose=0)
            mse = np.mean(np.power(X_scaled.values - reconstructed_data, 2), axis=1)
            predictions = (mse > threshold).astype(int)
        
        # Step 4: Finalization (100%)
        status_text.text("✅ Step 4/4: Finalizing results...")
        progress_bar.progress(100)
        
        return predictions, mse, df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        st.error(f"❌ Error during data processing: {e}")
        raise
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

# ==============================================================================
# Enhanced Visualization Functions
# ==============================================================================
def create_traffic_pie_chart(df_results: pd.DataFrame) -> go.Figure:
    """Create an enhanced donut chart for traffic distribution."""
    counts = df_results['Prediction'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[Config.COLORS['benign'], Config.COLORS['anomaly']],
        hole=0.4,  # Donut chart
        textinfo="label+percent+value",
        textfont=dict(size=12, color='white'),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title={
            'text': "Traffic Distribution",
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'family': 'Poppins'}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=60, b=20, l=20, r=20),
        height=400
    )
    
    return fig

def create_reconstruction_error_histogram(df_results: pd.DataFrame, threshold: float) -> go.Figure:
    """Create an enhanced histogram for reconstruction errors."""
    fig = px.histogram(
        df_results, 
        x="Reconstruction Error", 
        color="Prediction",
        color_discrete_map={
            'Benign': Config.COLORS['benign'], 
            'Anomaly': Config.COLORS['anomaly']
        },
        nbins=50,
        opacity=0.7
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold, 
        line_width=3, 
        line_dash="dash", 
        line_color=Config.COLORS['threshold'],
        annotation_text=f"Threshold: {threshold:.6f}",
        annotation_position="top right",
        annotation_font_color=Config.COLORS['threshold']
    )
    
    fig.update_layout(
        title={
            'text': f"Reconstruction Error Distribution",
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'family': 'Poppins'}
        },
        xaxis_title="Reconstruction Error",
        yaxis_title="Count",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(t=60, b=40, l=40, r=40),
        height=400
    )
    
    return fig

# ==============================================================================
# Main Application Logic
# ==============================================================================

# Load model components at startup
autoencoder, threshold, scaler = load_model_components()

# Sidebar with model information
with st.sidebar:
    st.markdown("### 🤖 Model Status")
    if autoencoder is not None:
        st.markdown('<p class="status-success">✅ Model loaded successfully</p>', unsafe_allow_html=True)
        st.info(f"**Threshold:** {threshold:.6f}")
        st.info(f"**Input Features:** {len(scaler.feature_names_in_) if scaler else 'N/A'}")
        st.info(f"**Model Type:** Deep Learning Autoencoder")
    else:
        st.markdown('<p class="status-error">❌ Model not loaded</p>', unsafe_allow_html=True)
        st.error("Please ensure model files are available")
    
    st.markdown("---")
    st.markdown("### 📁 File Requirements")
    st.markdown("""
    - **Format:** CSV file from CICFlowMeter
    - **Max Size:** 200MB
    - **Encoding:** UTF-8 recommended
    """)
    
    st.markdown("### ℹ️ How it Works")
    st.markdown("""
    1. **Upload** your network traffic CSV
    2. **Preprocessing** cleans and prepares data
    3. **Autoencoder** reconstructs normal patterns
    4. **Detection** flags high reconstruction errors as anomalies
    """)

# Main title and description
st.markdown("""
<div class="title-banner">
    <h1>Machine Learning Based Network Intrusion Detection System</h1>
</div>
<div class="title-description">
    <h3>Powered by a Deep Learning Autoencoder</h3>
    <p>Next-gen anomaly detection that spots malicious traffic by uncovering subtle deviations, redefining network security monitoring.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# File upload section
with st.container(border=True):
    st.subheader("📤 Upload Network Traffic Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file (from CICFlowMeter)",
        type="csv",
        help="Upload network traffic data in CSV format. Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        # File size validation
        if uploaded_file.size > Config.MAX_FILE_SIZE:
            st.error(f"❌ File too large! Maximum size allowed: {Config.MAX_FILE_SIZE / (1024*1024):.0f}MB")
            st.stop()
        
        # Model availability check
        if autoencoder is None:
            st.error("❌ Cannot process file: Model components not loaded properly.")
            st.stop()

# Main processing logic
if uploaded_file is not None and autoencoder is not None:
    try:
        # Load and validate data
        with st.spinner("📊 Loading data..."):
            df = pd.read_csv(uploaded_file)
        
        if not validate_dataframe(df):
            st.stop()
        
        # Display data overview
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / (1024*1024)
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
        st.success("✅ File uploaded and validated successfully!")
        
        # Process data
        st.markdown("---")
        st.subheader("🔍 Analysis Results")
        
        with st.spinner("🚀 Analyzing network traffic..."):
            predictions, mse, df_results = preprocess_and_predict(df.copy(), scaler, autoencoder, threshold)
        
        # Add results to dataframe
        df_results['Reconstruction Error'] = mse
        df_results['Prediction'] = np.where(predictions == 1, 'Anomaly', 'Benign')
        
        # Calculate metrics
        total_flows = len(df_results)
        num_anomalies = len(df_results[df_results['Prediction'] == 'Anomaly'])
        num_benign = total_flows - num_anomalies
        anomaly_percentage = (num_anomalies / total_flows) * 100 if total_flows > 0 else 0
        
        st.success("✅ Analysis completed successfully!")
        
        # Display summary metrics
        st.subheader("📊 Detection Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Flows Processed",
                value=f"{total_flows:,}",
                help="Total number of network flows analyzed"
            )
        with col2:
            st.metric(
                label="Benign Traffic",
                value=f"{num_benign:,}",
                delta=f"{100-anomaly_percentage:.1f}%",
                delta_color="normal"
            )
        with col3:
            st.metric(
                label="Anomalies Detected",
                value=f"{num_anomalies:,}",
                delta=f"{anomaly_percentage:.2f}%",
                delta_color="inverse"
            )
        with col4:
            threat_level = "🔴 High" if anomaly_percentage > 10 else "🟡 Medium" if anomaly_percentage > 1 else "🟢 Low"
            st.metric(
                label="Threat Level",
                value=threat_level,
                help="Based on anomaly percentage"
            )
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                fig_pie = create_traffic_pie_chart(df_results)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                fig_hist = create_reconstruction_error_histogram(df_results, threshold)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed anomaly report
        st.markdown("---")
        st.subheader("🚨 Detailed Analysis Report")
        
        if num_anomalies > 0:
            st.error(f"⚠️ **{num_anomalies} network flows flagged as ANOMALOUS**")
            
            # Show top anomalies by reconstruction error
            anomalies = df_results[df_results['Prediction'] == 'Anomaly'].copy()
            anomalies = anomalies.sort_values('Reconstruction Error', ascending=False)
            
            # Display anomalies with all feature variables
            st.markdown("**Detected Anomalies (with all feature variables):**")
            
            # Show all columns except internal processing columns
            columns_to_exclude = ['index']  # Only exclude true internal columns
            display_columns = [col for col in anomalies.columns if col not in columns_to_exclude]
            
            display_anomalies = anomalies[display_columns].head(100)  # Show top 100 anomalies
            st.dataframe(
                display_anomalies,
                use_container_width=True,
                column_config={
                    "Reconstruction Error": st.column_config.NumberColumn(
                        "Reconstruction Error",
                        format="%.6f"
                    )
                }
            )
            
            if len(anomalies) > 100:
                st.info(f"Showing top 100 anomalies out of {len(anomalies)} total.")
        else:
            st.success("✅ **No anomalies detected - Network traffic appears normal**")
            st.markdown("**Sample of analyzed traffic (with all feature variables):**")
            
            # Show all columns for normal traffic sample too
            columns_to_exclude = ['index']
            display_columns = [col for col in df_results.columns if col not in columns_to_exclude]
            sample_data = df_results[display_columns].head(20)
            
            st.dataframe(
                sample_data,
                use_container_width=True,
                column_config={
                    "Reconstruction Error": st.column_config.NumberColumn(
                        "Reconstruction Error",
                        format="%.6f"
                    )
                }
            )
            st.markdown("*(Showing first 20 flows)*")
        
        # Download section
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full report
            csv_report = df_results.to_csv(index=False)
            st.download_button(
                label="📄 Download Full Analysis Report",
                data=csv_report,
                file_name=f'ml_ids_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                help="Download complete dataset with predictions and reconstruction errors",
                use_container_width=True
            )
        
        with col2:
            # Anomalies only
            if num_anomalies > 0:
                anomalies_csv = df_results[df_results['Prediction'] == 'Anomaly'].to_csv(index=False)
                st.download_button(
                    label="⚠️ Download Anomalies Only",
                    data=anomalies_csv,
                    file_name=f'anomalies_only_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    help="Download only the detected anomalies",
                    use_container_width=True
                )
            else:
                st.info("No anomalies to export")
        
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty or corrupted.")
    except pd.errors.ParserError as e:
        st.error(f"❌ Unable to parse the CSV file: {e}")
        st.info("Please ensure the file is a valid CSV with proper formatting.")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        st.error(f"❌ An unexpected error occurred: {e}")
        st.info("Please check your data format and try again.")

elif uploaded_file is not None and autoencoder is None:
    st.warning("⚠️ Please ensure all model files are loaded before uploading data.")

# Footer section
st.markdown("---")
footer_html = f"""
<div style="text-align: center; margin-top: 3rem;">
    <div style="color: {theme_colors['text']}; font-size: 16px; margin-bottom: 15px; font-weight: 600;">
         Project by Dipankar Saha
    </div>
    <div class="social-icons" style="text-align: center; margin-bottom: 15px;">
        <a href="mailto:dsaha0427@gmail.com" title="Email">
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
    <div style="color: {theme_colors['text']}; font-size: 12px; opacity: 0.8;">
         <h5>Advanced Network Security •  Deep Learning Powered •  Real-time Threat Detection</h1>
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)