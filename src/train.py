import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

def train_model():
    print("--- Training UFC Fight Predictor Model (Temporal Split + Tuned XGBoost) ---")
    csv_path = os.path.join("data", "raw", "ufc-master.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    target_col = 'Winner'
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df_clean = df[df[target_col].isin(['red', 'blue'])].copy()
    
    df_clean['target'] = df_clean[target_col].map({'red': 1, 'blue': 0})

    # Ensure date column exists and sort chronologically for a true temporal split
    date_col = 'date' if 'date' in df_clean.columns else 'Date'
    if date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        print(f"📅 Sorting dataset chronologically by '{date_col}'...")
    else:
        print("⚠️ Warning: Date column not found. Falling back to index-based chronological split.")

    candidate_features = [
        'sig_str_dif', 'B_age', 'R_age', 'avg_td_dif', 'win_streak_dif',
        'loss_dif', 'B_avg_SIG_STR_landed', 'R_avg_SIG_STR_landed',
        'avg_sub_att_dif', 'age_dif', 'reach_dif', 'longest_win_streak_dif', 'height_dif',
        'elo_dif', 'ctrl_time_dif', 'knockdown_dif', 'SApM_dif', 
        'takedown_def_dif', 'layoff_dif', 'sig_str_last3_dif'
    ]

    features = [col for col in candidate_features if col in df_clean.columns]
    print(f"📊 Training on {len(features)} features:\n{features}")

    df_clean = df_clean.dropna(subset=features + ['target']).copy()

    X = df_clean[features]
    y = df_clean['target'].astype(int)

    # Strict Chronological / Temporal Split (First 80% train, Last 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📈 Training set: {len(X_train)} fights | Test set (Future): {len(X_test)} fights")

    # Tuned XGBoost Classifier Configuration
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.01,     # Slower, more precise learning rate
        max_depth=6,            # Slightly deeper trees for complex patterns
        subsample=0.8,          # Regularization
        colsample_bytree=0.8,   # Regularization
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Save Trained Model, Feature List, and Test Split
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, os.path.join("models", "ufc_model.pkl"))
    joblib.dump(features, os.path.join("models", "feature_names.pkl"))
    joblib.dump((X_test, y_test), os.path.join("models", "test_split.pkl"))
    
    print("✅ Temporal split model, feature list, and test split successfully saved to 'models/'!")

if __name__ == "__main__":
    train_model()