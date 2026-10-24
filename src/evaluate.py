import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, classification_report, confusion_matrix

def evaluate():
    print("--- Starting Phase 5: Enhanced Model Evaluation & SHAP Analysis ---")
    
    model_path = os.path.join("models", "ufc_model.pkl")
    features_path = os.path.join("models", "feature_names.pkl")
    test_split_path = os.path.join("models", "test_split.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(test_split_path):
        print("❌ Trained model or test split not found. Run train.py first.")
        return

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    X_test, y_test = joblib.load(test_split_path)

    y_test = y_test.astype(int)

    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Benchmark Metrics
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print("\n" + "="*50)
    print("MODEL EVALUATION & BENCHMARK METRICS")
    print("="*50)
    print(f"• Accuracy:    {acc*100:.2f}%")
    print(f"• Log Loss:    {loss:.4f}")
    print(f"• ROC-AUC:     {auc:.4f}")
    print(f"• Brier Score: {brier:.4f} (0.0 = perfect calibration)")
    print("="*50 + "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Blue Corner Wins", "Red Corner Wins"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # SHAP Feature Importance Analysis
    print("\nCalculating SHAP feature importance values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(axis=0)
    elif len(shap_values.shape) == 3:
        vals = np.abs(shap_values[:, :, 1]).mean(axis=0)
    else:
        vals = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({'Feature': features, 'Mean |SHAP|': vals})
    feature_importance = feature_importance.sort_values(by='Mean |SHAP|', ascending=False)

    print("\nTop Features Driving Predictions:")
    print(feature_importance.to_string(index=False))

    # Save SHAP Summary Plot
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "shap_summary.png"))
    plt.close()
    print("\n✅ SHAP summary plot saved to 'reports/shap_summary.png'")

if __name__ == "__main__":
    evaluate()