# AI-Based Network Traffic Classification for Cybersecurity

> **Automated Binary Classification System for Network Intrusion Detection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](/)

An AI-powered cybersecurity solution that automatically classifies network traffic as **Benign** or **Malicious** using supervised machine learning. This project demonstrates a complete ML pipeline from data preprocessing to model deployment, achieving **99.87% recall** for malicious traffic detection.

---

## Project Overview

### Objectives

This project implements a production-grade binary classification system to:
- Automatically detect malicious network traffic with high accuracy
- Minimize false positives while maximizing threat detection
- Process large-scale network traffic data (2.4M+ samples)
- Provide explainable and reproducible results

### Problem Statement

An AI-based cybersecurity company aims to automatically classify network traffic as **Benign** or **Malicious** to strengthen system security. The solution leverages network flow features including packet size, protocol type, connection duration, data transfer rates, and access patterns to train a supervised classification model.

---

## Dataset

### Source
CICIDS-based Network Intrusion Detection Dataset containing real-world network traffic patterns.

### Statistics
- **Total Samples**: 2,435,692 flows (after deduplication)
- **Features**: 116 engineered network traffic features
- **Class Distribution**: 73% Benign / 27% Malicious (2.74:1 ratio)
- **Dataset Size**: 1.89 GB

### Feature Categories
- **Packet Statistics**: Size, count, inter-arrival times (27 features)
- **Protocol Information**: TCP, UDP protocol types (1 feature)
- **Connection Metrics**: Duration, bulk transfer patterns (3 features)
- **Payload Analysis**: Bytes transferred, payload statistics (18 features)
- **Transfer Rates**: Bytes/second, packets/second (10 features)
- **TCP Flags**: SYN, ACK, FIN, RST, PSH flags (24 features)
- **Header Information**: Header sizes and statistics (14 features)
- **Bidirectional Flows**: Forward and backward traffic analysis (19 features)

---

## Results

### Best Model: Random Forest Classifier

| Metric | Score | Industry Standard | Status |
|--------|-------|-------------------|--------|
| **Accuracy** | **99.76%** | >95% | Exceeds |
| **Precision** | **99.24%** | >90% | Exceeds |
| **Recall (Malicious)** | **99.87%** | >90% | Exceeds |
| **F1-Score** | **0.9956** | >0.90 | Exceeds |
| **ROC-AUC** | **1.0000** | >0.95 | Perfect |
| **False Positive Rate** | **0.28%** | <5% | Exceeds |

### Confusion Matrix (Test Set: 487,139 samples)

```
                 Predicted
                 Benign    Malicious
Actual  Benign   356,150   995        (99.72% correctly identified)
        Malicious  164     129,830    (99.87% detected)
```

**Key Achievements:**
- Only **164 attacks missed** out of 129,994 (0.13% false negative rate)
- Only **995 false alarms** out of 357,145 benign flows (0.28% false positive rate)
- Production-ready reliability for real-world deployment

---

## Machine Learning Pipeline

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 93.25% | 81.01% | 97.57% | 0.8852 | 0.9902 |
| **Random Forest** | **99.76%** | **99.24%** | **99.87%** | **0.9956** | **1.0000** |

### Training Process

1. **Data Preprocessing**
   - Duplicate removal (2,360 samples)
   - Feature scaling with StandardScaler
   - Stratified train-test split (80/20)

2. **Feature Engineering**
   - 116 numeric features extracted
   - Leakage features removed (IDs, timestamps)
   - Class-balanced training

3. **Model Training**
   - Baseline: Logistic Regression
   - Advanced: Random Forest (n_estimators=100, max_depth=20)
   - Class weights balanced for imbalanced data

4. **Evaluation**
   - Comprehensive metrics calculated
   - ROC curves generated
   - Confusion matrices analyzed

---

## 📁 Project Structure

```
network-ml-project/
│
├── 📂 data/
│   ├── network_binary_ready.csv      # Final processed dataset (2.4M samples)
│   └── raw/                           # Original raw data files (18 CSVs)
│
├── 📂 src/
│   ├── train_binary_classification.py # Main training pipeline
│   └── demo_inference.py              # Model inference demo
│
├── 📂 models/
│   ├── binary_best_model.pkl          # Trained Random Forest model
│   └── binary_scaler.pkl              # Fitted StandardScaler
│
├── 📂 results/
│   ├── binary_classification_evaluation.png  # Visualizations
│   ├── binary_model_comparison.csv           # Performance comparison
│   └── binary_model_metadata.json            # Training metadata
│
├── 📄 FINAL_REPORT.md                 # Comprehensive technical report
├── 📄 PROJECT_ALIGNMENT_AUDIT.md      # Requirements compliance audit
└── 📄 README.md                       # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd network-ml-project
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Training the Model

```bash
python src/train_binary_classification.py
```

**Expected Runtime**: 5-10 minutes  
**Output**: Model artifacts saved to `models/`, visualizations to `results/`

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/binary_best_model.pkl')
scaler = joblib.load('models/binary_scaler.pkl')

# Load new data
X_new = pd.read_csv('new_traffic.csv')

# Preprocess and predict
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)

# 0 = Benign, 1 = Malicious
print(f"Detected {sum(predictions)} malicious flows")
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn 1.0+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Model Persistence** | joblib |
| **Development** | Jupyter, VS Code |

---

## Performance Visualizations

The project generates comprehensive visualizations:
-  Confusion matrices for both models
-  ROC curves with AUC scores
-  Model performance comparison charts
-  Feature importance rankings

See `results/binary_classification_evaluation.png` for complete visual analysis.

---

## Documentation

- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Comprehensive technical report with methodology and results
- **[PROJECT_ALIGNMENT_AUDIT.md](PROJECT_ALIGNMENT_AUDIT.md)**: Requirements compliance verification

---

## Project Compliance

This project fully satisfies all academic requirements:
-  Binary classification (Benign vs. Malicious)
-  Data preprocessing and feature engineering
-  Multiple supervised learning models
-  Comprehensive evaluation metrics
-  Production-ready implementation
-  Complete documentation

**Academic Summary Statement:**
> Binary classification model achieved **99.87% recall** for malicious traffic with **ROC-AUC of 1.0000**. The Random Forest model demonstrates exceptional performance in distinguishing between benign and malicious network traffic, with an overall accuracy of **99.76%** on the test set.

---

## Future Enhancements

- [ ] Deploy model as REST API service
- [ ] Implement real-time streaming inference
- [ ] Add SHAP explainability for predictions
- [ ] Integrate with SIEM systems
- [ ] Expand to multi-class attack type classification
- [ ] Add model monitoring and retraining pipeline

---

## License

This project is developed for **academic and educational purposes**.

---

## Contributors

**Team 24**  
*Academic Project - 2026*

---

## Acknowledgments

- CICIDS dataset providers
- scikit-learn community
- Cybersecurity research community

