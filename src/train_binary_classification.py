"""
BINARY CLASSIFICATION TRAINING PIPELINE
Cybersecurity Network Traffic Classification: Benign vs. Malicious

This script implements a complete binary classification pipeline suitable for 
academic submission, including model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ================================================================================
# CONFIGURATION
# ================================================================================

print("=" * 100)
print("BINARY CLASSIFICATION TRAINING PIPELINE")
print("Network Traffic: Benign vs. Malicious")
print("=" * 100)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "network_binary_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# ================================================================================
# TASK 1 — DATA LOADING AND PREPARATION
# ================================================================================

print("\n" + "=" * 100)
print("TASK 1 — DATA LOADING AND PREPARATION")
print("=" * 100)

print(f"\n📂 Loading dataset: {DATA_FILE.name}")
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

# Show initial shape
print(f"\n📊 Initial Dataset Shape: {df.shape}")

# Drop duplicates if present
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)
if duplicates_removed > 0:
    print(f"🗑️  Removed {duplicates_removed:,} duplicate rows ({duplicates_removed/initial_rows*100:.2f}%)")
else:
    print(f"✅ No duplicate rows found")

print(f"\n📊 Final Dataset Shape: {df.shape}")

# Identify target and features
print(f"\n🎯 Target Variable: 'binary_label'")
print(f"   Class 0 (Benign): {(df['binary_label'] == 0).sum():,} samples")
print(f"   Class 1 (Malicious): {(df['binary_label'] == 1).sum():,} samples")

# Drop non-predictive columns
columns_to_drop = ['label', 'binary_label', 'binary_label_text']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

print(f"\n🗑️  Dropping non-predictive columns: {columns_to_drop}")

# Separate features and target
X = df.drop(columns=columns_to_drop, errors='ignore')
y = df['binary_label']

# Keep only numeric features
initial_features = X.shape[1]
X = X.select_dtypes(include=[np.number])
features_kept = X.shape[1]

print(f"\n📋 Feature Engineering:")
print(f"   Initial features: {initial_features}")
print(f"   Numeric features: {features_kept}")
print(f"   Non-numeric dropped: {initial_features - features_kept}")

# Train-test split
print(f"\n✂️  Splitting data (80% train / 20% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"✅ Train set: {X_train.shape[0]:,} samples")
print(f"✅ Test set:  {X_test.shape[0]:,} samples")
print(f"\n   Train class distribution:")
print(f"      Benign (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"      Malicious (1): {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"\n   Test class distribution:")
print(f"      Benign (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"      Malicious (1): {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

# ================================================================================
# TASK 2 — FEATURE SCALING
# ================================================================================

print("\n" + "=" * 100)
print("TASK 2 — FEATURE SCALING")
print("=" * 100)

print(f"\n⚙️  Applying StandardScaler...")
scaler = StandardScaler()

# Fit on training data only
scaler.fit(X_train)
print(f"✅ Scaler fitted on training data")

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ Training data scaled: {X_train_scaled.shape}")
print(f"✅ Test data scaled: {X_test_scaled.shape}")

# Save scaler
scaler_path = MODELS_DIR / "binary_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"💾 Scaler saved: {scaler_path}")

# ================================================================================
# TASK 3 — TRAIN BASELINE MODEL (LOGISTIC REGRESSION)
# ================================================================================

print("\n" + "=" * 100)
print("TASK 3 — TRAIN BASELINE MODEL (LOGISTIC REGRESSION)")
print("=" * 100)

print(f"\n🔧 Training Logistic Regression...")
print(f"   Settings:")
print(f"      max_iter = 1000")
print(f"      class_weight = 'balanced'")
print(f"      random_state = {RANDOM_STATE}")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lr_model.fit(X_train_scaled, y_train)
print(f"✅ Logistic Regression trained successfully")

# Make predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"✅ Predictions generated")

# ================================================================================
# TASK 4 — TRAIN STRONG MODEL (RANDOM FOREST)
# ================================================================================

print("\n" + "=" * 100)
print("TASK 4 — TRAIN STRONG MODEL (RANDOM FOREST)")
print("=" * 100)

print(f"\n🔧 Training Random Forest Classifier...")
print(f"   Settings:")
print(f"      n_estimators = 100")
print(f"      max_depth = 20")
print(f"      class_weight = 'balanced'")
print(f"      random_state = {RANDOM_STATE}")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)
print(f"✅ Random Forest trained successfully")

# Make predictions
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)
rf_test_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print(f"✅ Predictions generated")

# ================================================================================
# TASK 5 — MODEL EVALUATION
# ================================================================================

print("\n" + "=" * 100)
print("TASK 5 — MODEL EVALUATION")
print("=" * 100)

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, pos_label=1),
        'Recall': recall_score(y_true, y_pred, pos_label=1),
        'F1_Score': f1_score(y_true, y_pred, pos_label=1),
        'ROC_AUC': roc_auc_score(y_true, y_pred_proba),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

# Evaluate Logistic Regression
print("\n📊 LOGISTIC REGRESSION EVALUATION")
print("-" * 100)
lr_metrics = evaluate_model(y_test, lr_test_pred, lr_test_pred_proba, "Logistic Regression")

print(f"\n🎯 Test Set Metrics:")
print(f"   Accuracy:  {lr_metrics['Accuracy']:.4f} ({lr_metrics['Accuracy']*100:.2f}%)")
print(f"   Precision: {lr_metrics['Precision']:.4f} ({lr_metrics['Precision']*100:.2f}%)")
print(f"   Recall (Malicious): {lr_metrics['Recall']:.4f} ({lr_metrics['Recall']*100:.2f}%)")
print(f"   F1 Score:  {lr_metrics['F1_Score']:.4f}")
print(f"   ROC AUC:   {lr_metrics['ROC_AUC']:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, lr_test_pred, target_names=['Benign', 'Malicious'], digits=4))

print(f"\n📊 Confusion Matrix:")
cm = lr_metrics['Confusion_Matrix']
print(f"   True Negatives (TN):  {cm[0,0]:,}")
print(f"   False Positives (FP): {cm[0,1]:,}")
print(f"   False Negatives (FN): {cm[1,0]:,}")
print(f"   True Positives (TP):  {cm[1,1]:,}")

# Evaluate Random Forest
print("\n" + "-" * 100)
print("📊 RANDOM FOREST EVALUATION")
print("-" * 100)
rf_metrics = evaluate_model(y_test, rf_test_pred, rf_test_pred_proba, "Random Forest")

print(f"\n🎯 Test Set Metrics:")
print(f"   Accuracy:  {rf_metrics['Accuracy']:.4f} ({rf_metrics['Accuracy']*100:.2f}%)")
print(f"   Precision: {rf_metrics['Precision']:.4f} ({rf_metrics['Precision']*100:.2f}%)")
print(f"   Recall (Malicious): {rf_metrics['Recall']:.4f} ({rf_metrics['Recall']*100:.2f}%)")
print(f"   F1 Score:  {rf_metrics['F1_Score']:.4f}")
print(f"   ROC AUC:   {rf_metrics['ROC_AUC']:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, rf_test_pred, target_names=['Benign', 'Malicious'], digits=4))

print(f"\n📊 Confusion Matrix:")
cm = rf_metrics['Confusion_Matrix']
print(f"   True Negatives (TN):  {cm[0,0]:,}")
print(f"   False Positives (FP): {cm[0,1]:,}")
print(f"   False Negatives (FN): {cm[1,0]:,}")
print(f"   True Positives (TP):  {cm[1,1]:,}")

# ================================================================================
# TASK 6 — VISUALIZATION
# ================================================================================

print("\n" + "=" * 100)
print("TASK 6 — VISUALIZATION")
print("=" * 100)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# -------------------- Confusion Matrix: Logistic Regression --------------------
ax1 = axes[0, 0]
cm_lr = lr_metrics['Confusion_Matrix']
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=True,
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
ax1.set_title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

# -------------------- Confusion Matrix: Random Forest --------------------
ax2 = axes[0, 1]
cm_rf = rf_metrics['Confusion_Matrix']
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=True,
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
ax2.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)

# -------------------- ROC Curve Comparison --------------------
ax3 = axes[1, 0]

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_pred_proba)
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax3.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
         label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_pred_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax3.plot(fpr_rf, tpr_rf, color='green', lw=2,
         label=f'Random Forest (AUC = {roc_auc_rf:.4f})')

# Diagonal reference line
ax3.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')

ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(True, alpha=0.3)

# -------------------- Metrics Comparison Bar Chart --------------------
ax4 = axes[1, 1]

metrics_comparison = {
    'Accuracy': [lr_metrics['Accuracy'], rf_metrics['Accuracy']],
    'Precision': [lr_metrics['Precision'], rf_metrics['Precision']],
    'Recall': [lr_metrics['Recall'], rf_metrics['Recall']],
    'F1 Score': [lr_metrics['F1_Score'], rf_metrics['F1_Score']],
    'ROC AUC': [lr_metrics['ROC_AUC'], rf_metrics['ROC_AUC']]
}

x = np.arange(len(metrics_comparison))
width = 0.35

lr_values = [lr_metrics['Accuracy'], lr_metrics['Precision'], lr_metrics['Recall'], 
             lr_metrics['F1_Score'], lr_metrics['ROC_AUC']]
rf_values = [rf_metrics['Accuracy'], rf_metrics['Precision'], rf_metrics['Recall'], 
             rf_metrics['F1_Score'], rf_metrics['ROC_AUC']]

lr_bars = ax4.bar(x - width/2, lr_values, width, label='Logistic Regression', color='skyblue')
rf_bars = ax4.bar(x + width/2, rf_values, width, label='Random Forest', color='lightgreen')

ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_comparison.keys(), rotation=45, ha='right')
ax4.legend()
ax4.set_ylim([0, 1.1])
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (metric_name, values) in enumerate(metrics_comparison.items()):
    ax4.text(i - width/2, values[0] + 0.02, f'{values[0]:.3f}', 
             ha='center', va='bottom', fontsize=9)
    ax4.text(i + width/2, values[1] + 0.02, f'{values[1]:.3f}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save figure
viz_path = RESULTS_DIR / "binary_classification_evaluation.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"\n💾 Visualization saved: {viz_path}")

print(f"✅ Generated:")
print(f"   - Confusion matrices (both models)")
print(f"   - ROC curves (comparison)")
print(f"   - Metrics comparison bar chart")

# ================================================================================
# TASK 7 — MODEL COMPARISON TABLE
# ================================================================================

print("\n" + "=" * 100)
print("TASK 7 — MODEL COMPARISON TABLE")
print("=" * 100)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_metrics['Accuracy'], rf_metrics['Accuracy']],
    'Precision': [lr_metrics['Precision'], rf_metrics['Precision']],
    'Recall': [lr_metrics['Recall'], rf_metrics['Recall']],
    'F1_Score': [lr_metrics['F1_Score'], rf_metrics['F1_Score']],
    'ROC_AUC': [lr_metrics['ROC_AUC'], rf_metrics['ROC_AUC']]
})

print("\n📊 MODEL PERFORMANCE COMPARISON")
print("-" * 100)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_path = RESULTS_DIR / "binary_model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"\n💾 Comparison table saved: {comparison_path}")

# ================================================================================
# TASK 8 — SAVE BEST MODEL
# ================================================================================

print("\n" + "=" * 100)
print("TASK 8 — SAVE BEST MODEL")
print("=" * 100)

print(f"\n🏆 Selecting Best Model...")
print(f"   Selection Criteria:")
print(f"      1. Highest Malicious Recall (Priority)")
print(f"      2. Highest F1 Score (Tiebreaker)")

# Compare models
if lr_metrics['Recall'] > rf_metrics['Recall']:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_metrics = lr_metrics
elif rf_metrics['Recall'] > lr_metrics['Recall']:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_metrics = rf_metrics
else:
    # Tie in recall, use F1 score
    if rf_metrics['F1_Score'] >= lr_metrics['F1_Score']:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_metrics = lr_metrics

print(f"\n✅ Best Model: {best_model_name}")
print(f"   Recall (Malicious): {best_metrics['Recall']:.4f} ({best_metrics['Recall']*100:.2f}%)")
print(f"   F1 Score: {best_metrics['F1_Score']:.4f}")
print(f"   ROC AUC: {best_metrics['ROC_AUC']:.4f}")

# Save best model
model_path = MODELS_DIR / "binary_best_model.pkl"
joblib.dump(best_model, model_path)
print(f"\n💾 Best model saved: {model_path}")
print(f"💾 Scaler saved: {scaler_path}")

# Save metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': float(best_metrics['Accuracy']),
    'precision': float(best_metrics['Precision']),
    'recall': float(best_metrics['Recall']),
    'f1_score': float(best_metrics['F1_Score']),
    'roc_auc': float(best_metrics['ROC_AUC']),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_features': X_train.shape[1],
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

import json
metadata_path = RESULTS_DIR / "binary_model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"💾 Metadata saved: {metadata_path}")

# ================================================================================
# TASK 9 — ACADEMIC REPORT OUTPUT
# ================================================================================

print("\n" + "=" * 100)
print("TASK 9 — ACADEMIC REPORT SUMMARY")
print("=" * 100)

print(f"\n🎓 FINAL SUMMARY FOR ACADEMIC SUBMISSION")
print("-" * 100)

print(f"\n📊 Dataset Statistics:")
print(f"   Total samples: {len(df):,}")
print(f"   Training samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")
print(f"   Number of features: {X_train.shape[1]}")
print(f"   Class distribution: {(y==0).sum():,} Benign / {(y==1).sum():,} Malicious")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"\n🎯 Key Performance Metrics:")
print(f"   ✓ Accuracy:  {best_metrics['Accuracy']*100:.2f}%")
print(f"   ✓ Precision: {best_metrics['Precision']*100:.2f}%")
print(f"   ✓ Recall (Malicious): {best_metrics['Recall']*100:.2f}%")
print(f"   ✓ F1 Score:  {best_metrics['F1_Score']:.4f}")
print(f"   ✓ ROC AUC:   {best_metrics['ROC_AUC']:.4f}")

print(f"\n📝 ACADEMIC SUMMARY STATEMENT:")
print("-" * 100)
print(f"\n   Binary classification model achieved {best_metrics['Recall']*100:.2f}% recall")
print(f"   for malicious traffic with ROC-AUC of {best_metrics['ROC_AUC']:.4f}.")
print(f"\n   The {best_model_name} model demonstrates strong performance in")
print(f"   distinguishing between benign and malicious network traffic, with")
print(f"   an overall accuracy of {best_metrics['Accuracy']*100:.2f}% on the test set.")
print("-" * 100)

print(f"\n📁 Output Files Created:")
print(f"   ✓ {model_path}")
print(f"   ✓ {scaler_path}")
print(f"   ✓ {viz_path}")
print(f"   ✓ {comparison_path}")
print(f"   ✓ {metadata_path}")

print(f"\n" + "=" * 100)
print(f"✅ BINARY CLASSIFICATION PIPELINE COMPLETE")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
