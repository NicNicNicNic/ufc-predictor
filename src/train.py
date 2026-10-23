import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train():
    print("--- Starting Training Pipeline ---")
    
    data_path = os.path.join("data", "raw", "ufc-master.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Successfully loaded dataset with shape: {df.shape}")
    
    # Filter out draws (keep only Red or Blue winners)
    print("Filtering out draws and mapping target...")
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    df['target'] = (df['Winner'] == 'Red').astype(int)
    
    # Selected features
    feature_cols = [
        'reach_dif', 'age_dif', 'height_dif', 
        'win_streak_dif', 'loss_dif', 'longest_win_streak_dif',
        'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif',
        'R_age', 'B_age', 'R_avg_SIG_STR_landed', 'B_avg_SIG_STR_landed'
    ]
    
    print("Cleaning dataset (dropping missing values in feature columns)...")
    clean_df = df[feature_cols + ['target']].dropna()
    print(f"Cleaned dataset shape: {clean_df.shape}")
    
    X = clean_df[feature_cols]
    y = clean_df['target']
    
    # Split data into 80% training and 20% testing sets
    print("Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize/Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier (this may take a few seconds)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train_scaled, y_train)
    
    # Save trained model and test data to models/ folder
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ufc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, scaler, X_test, y_test), f)
        
    print(f"✅ Success! Model successfully saved to '{model_path}'!")

if __name__ == "__main__":
    train()