import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, classification_report, confusion_matrix

def evaluate():
    print("--- Starting Phase 5: Enhanced Model Evaluation & SHAP Analysis ---")
    
    model_path = os.path.join("models", "ufc_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please run 'python src/train.py' first.")
        
    # Load trained model, scaler, and test set
    with open(model_path, "rb") as f:
        model, scaler, X_test, y_test = pickle.load(f)
        
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict outcomes and probabilities
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    
    print("\n" + "=" * 45)
    print("   MODEL EVALUATION & BENCHMARK METRICS")
    print("=" * 45)
    print(f"• Accuracy:     {acc * 100:.2f}%")
    print(f"• Log Loss:     {loss:.4f}")
    print(f"• ROC-AUC:      {auc:.4f}")
    print(f"• Brier Score:  {brier:.4f}  (0.0 = perfect calibration)")
    print("=" * 45)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Blue Corner Wins", "Red Corner Wins"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # --- SHAP Feature Importance Analysis ---
    print("\nCalculating SHAP feature importance values...")
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test_scaled)
    
    # Extract Class 1 (Red Corner Win) SHAP matrix
    if isinstance(raw_shap, list):
        shap_vals = np.array(raw_shap[1])
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            shap_vals = raw_shap[:, :, 1]
        else:
            shap_vals = raw_shap
    else:
        # Fallback if raw_shap is an Explanation object
        shap_vals = np.array(raw_shap.values)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

    # Calculate 1D mean absolute SHAP value per feature
    feature_names = list(X_test.columns)
    mean_shap = np.abs(shap_vals).mean(axis=0).flatten()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP| Value': mean_shap
    }).sort_values(by='Mean |SHAP| Value', ascending=False)
    
    print("\nTop Features Driving Predictions:")
    print(importance_df.to_string(index=False))
    
    # Generate and save SHAP summary plot
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plot_path = os.path.join("reports", "shap_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✅ SHAP summary plot saved to '{plot_path}'")

if __name__ == "__main__":
    evaluate()