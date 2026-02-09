# AI-Based Network Traffic Classification for Cybersecurity

## 📌 Project Overview

This project focuses on building a Machine Learning model to classify network traffic as **Benign** or **Malicious**.
The goal is to automate threat detection using supervised learning on structured network traffic data.

The project is designed as an academic ML pipeline with real-world cybersecurity relevance.

---

## 🎯 Objectives

* Preprocess raw network traffic dataset
* Perform feature encoding and scaling
* Train classification models
* Evaluate models using security-focused metrics
* Identify best performing model for intrusion detection

---

## 📂 Dataset

Dataset Used:

* Network Intrusion Detection Dataset (Kaggle)

Type:

* Tabular network traffic data

Contains:

* Packet size
* Protocol type
* Connection duration
* Access patterns
* Traffic behavior features
* Label column (Benign / Malicious)

---

## 🧱 Project Pipeline

```
Raw Dataset
   ↓
Data Cleaning
   ↓
Encoding + Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Final Model Selection
```

---

## 🧠 Machine Learning Models Used

### 1️⃣ Logistic Regression

* Used as baseline model
* Fast training
* Good interpretability

### 2️⃣ Random Forest

* Main model candidate
* Handles non-linear relationships
* Provides feature importance
* Usually stronger on tabular security data

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall (Critical for detecting attacks)
* F1 Score
* Confusion Matrix

---

## 🧰 Tech Stack

### Language

* Python 3

### Libraries

* pandas → Data handling
* numpy → Numerical operations
* scikit-learn → Machine learning models
* matplotlib → Visualization
* seaborn → Advanced visualization
* joblib → Model saving

---

## 📁 Project Structure

```
network-ml-project/
│
├── data/
│   └── network.csv
│
├── src/
│   └── train_baseline.py
│
├── models/
│
├── results/
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install Python

Download from:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

---

### 2️⃣ Install Required Libraries

Run in Command Prompt:

```
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

### 3️⃣ Run Training Script

Navigate to source folder:

```
cd src
python train_baseline.py
```

---

## ✅ Current Progress

✔ Python Environment Setup Complete
✔ Required ML Libraries Installed
✔ Dataset Selected
✔ Project Structure Created
✔ Baseline Training Script Ready

---

## 🚀 Next Steps

* Train Logistic Regression baseline model
* Train Random Forest model
* Compare performance metrics
* Generate confusion matrix
* Finalize best model
* Update PPT with real results

---

## 🔒 Scope Limitations

This project does NOT include:

* Deep Learning models
* Real-time packet capture
* IDS/IPS deployment integration
* Streaming data processing

Focus is on supervised ML classification pipeline.

---

## 👥 Team

[Add Group Member Names]

---

## 📜 License

Academic / Educational Use
