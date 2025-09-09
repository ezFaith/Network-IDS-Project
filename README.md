# Machine Learning-Based Network Intrusion Detection System (IDS)

## ✦︎ Project Overview
This project is a **cybersecurity application** that leverages a **Deep Learning Autoencoder** for **unsupervised anomaly detection** in network traffic. Unlike traditional signature-based IDS solutions, this system learns the statistical patterns of *normal traffic* and proactively flags deviations as potential intrusions — including **zero-day attacks**.

The IDS is deployed as a **Streamlit web application**, providing an interactive interface for **data upload, analysis visualization, and report generation**.

---

## ✦︎ Key Features

- **Deep Learning Model** – TensorFlow-based autoencoder trained exclusively on benign traffic for unsupervised detection.  
- **Zero-Day Threat Detection** – Identifies unknown threats without relying on signatures or labeled attack data.  
- **End-to-End Pipeline** – From packet capture and feature extraction to anomaly detection and web deployment.  
- **Streamlit Web UI** – Upload CSV files, run instant predictions, and view results in an intuitive dashboard.  
- **Interactive Visualizations** – Includes:  
  - Traffic distribution pie chart  
  - Reconstruction error histogram  
- **Downloadable Reports** – Export enriched CSVs containing predictions and anomaly scores.  

---

## ✦︎ Getting Started

### Prerequisites
Ensure you have the following installed:  
- **Python 3.8+**  
- **Java Runtime Environment (JRE)** (for CICFlowMeter)  
- **Git**  
- **Wireshark (with tshark.exe)** for packet capture & feature extraction  

### Installation

```bash
# Clone the repository
git clone https://github.com/ezFaith/Network-IDS-Project.git
cd Network-IDS-Project

# Set up virtual environment
python -m venv venv
# Activate venv (Windows)
.\venv\Scripts\activate
# Activate venv (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Model Files
Due to file size limits, model artifacts are not included. Place the following in the project root:  
- `autoencoder_model.h5`  
- `scaler.pkl`  
- `threshold.pkl`  

### Run the Web App

```bash
streamlit run app.py
```

Application will be available at: **http://localhost:8501**  

---

## ✦︎ Methodology

### Data & Feature Engineering
- Training dataset: **CICIDS2017**  
- Raw traffic captured via **Wireshark**  
- Feature extraction using **CICFlowMeter**  
- Data aligned with training schema for consistency  

### Model Architecture
- Deep Autoencoder:  
  `79 → 64 → 32 → 16 → 8 → 16 → 32 → 64 → 79`  
- Trained on **benign traffic only**  
- Optimized for **minimal reconstruction loss**  

### Detection Logic
- Calculate **Mean Squared Error (MSE)** between input and reconstruction  
- Flag traffic flows with error > threshold as anomalies  

---

## ✦︎ Live Demo
Try the hosted application here:  
» [Streamlit App](https://projectnids.streamlit.app)  

---

## 📧 Contact
**Project by:** Dipankar Saha  

- 📩 Email: [dsaha0427@gmail.com](mailto:dsaha0427@gmail.com)  
- 💼 LinkedIn: [linkedin.com/in/dipankarsaha2001](https://www.linkedin.com/in/dipankarsaha2001/)  
- 🌐 Portfolio: [ezfaith.github.io/portfolio](https://ezfaith.github.io/portfolio/)  
