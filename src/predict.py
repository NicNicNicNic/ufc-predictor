import os
import pandas as pd
import numpy as np
import joblib

def load_model_and_artifacts():
    model_path = os.path.join("models", "ufc_model.pkl")
    features_path = os.path.join("models", "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("❌ Model or feature list not found! Please run train.py first.")
        return None, None
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def build_global_fighter_registry(df):
    registry = {}
    r_cols = ['R_fighter', 'R_age', 'R_Reach_cms', 'R_Height_cms', 'R_avg_SIG_STR_landed', 'R_current_win_streak', 'weight_class']
    r_df = df[[c for c in r_cols if c in df.columns]].copy()
    r_df.columns = ['fighter', 'age', 'reach', 'height', 'sig_str_landed', 'win_streak', 'weight_class'][:r_df.shape[1]]
    
    b_cols = ['B_fighter', 'B_age', 'B_Reach_cms', 'B_Height_cms', 'B_avg_SIG_STR_landed', 'B_current_win_streak', 'weight_class']
    b_df = df[[c for c in b_cols if c in df.columns]].copy()
    b_df.columns = ['fighter', 'age', 'reach', 'height', 'sig_str_landed', 'win_streak', 'weight_class'][:b_df.shape[1]]
    
    combined = pd.concat([r_df, b_df]).dropna(subset=['fighter'])
    latest = combined.drop_duplicates(subset=['fighter'], keep='last')
    
    for _, row in latest.iterrows():
        name_key = str(row['fighter']).strip().lower()
        registry[name_key] = {
            'name': str(row['fighter']),
            'age': row.get('age', df['R_age'].median()),
            'reach': row.get('reach', df['R_Reach_cms'].median()),
            'height': row.get('height', df['R_Height_cms'].median()),
            'sig_str_landed': row.get('sig_str_landed', df['R_avg_SIG_STR_landed'].median()),
            'win_streak': row.get('win_streak', 0),
            'weight_class': row.get('weight_class', 'Unknown')
        }
        
    # Manual safety patches for prominent names / recent cross-promotion additions
    registry['patricio pitbull'] = {
        'name': 'Patricio Pitbull', 'age': 39, 'reach': 67.0, 'height': 66.0, 
        'sig_str_landed': 38.0, 'win_streak': 1, 'weight_class': 'Featherweight'
    }
    registry['doo-ho choi'] = {
        'name': 'Doo-Ho Choi', 'age': 35, 'reach': 70.0, 'height': 70.0, 
        'sig_str_landed': 20.0, 'win_streak': 0, 'weight_class': 'Featherweight'
    }
    registry['dooho choi'] = registry['doo-ho choi']
    
    return registry

def find_fighter_match(query, registry):
    query = query.strip().lower()
    if not query:
        return []
    
    matches = []
    for key, data in registry.items():
        full_name = data.get('name', '').lower()
        if query in key or query in full_name:
            if data not in matches:
                matches.append(data)
                
    return matches[:10]

def select_fighter(prompt_text, registry, df):
    while True:
        user_input = input(prompt_text).strip()
        if user_input.lower() in ['q', 'exit']:
            return None
            
        matches = find_fighter_match(user_input, registry)
        
        if len(matches) == 0:
            print(f"\n⚠️ '{user_input}' not found in local records.")
            choice = input("Would you like to create a custom profile for them using division averages? (y/n): ").strip().lower()
            if choice == 'y':
                return {
                    'name': user_input.title(),
                    'age': 29.0,
                    'reach': df['R_Reach_cms'].median(),
                    'height': df['R_Height_cms'].median(),
                    'sig_str_landed': df['R_avg_SIG_STR_landed'].median(),
                    'win_streak': 1,
                    'weight_class': 'Custom/Cross-Promotion'
                }
            else:
                continue
                
        elif len(matches) == 1:
            selected = matches[0]
            print(f"✅ Selected: {selected['name']}")
            return selected
        else:
            print(f"\n🔍 Multiple matches found for '{user_input}':")
            for idx, item in enumerate(matches, 1):
                print(f"  [{idx}] {item['name']} ({item['weight_class']})")
            
            choice = input("Enter number of the correct fighter (or type 'new' to create custom): ").strip()
            if choice.lower() == 'new':
                return {
                    'name': user_input.title(),
                    'age': 29.0,
                    'reach': df['R_Reach_cms'].median(),
                    'height': df['R_Height_cms'].median(),
                    'sig_str_landed': df['R_avg_SIG_STR_landed'].median(),
                    'win_streak': 1,
                    'weight_class': 'Custom/Cross-Promotion'
                }
            elif choice.isdigit() and 1 <= int(choice) <= len(matches):
                selected = matches[int(choice) - 1]
                print(f"✅ Selected: {selected['name']}")
                return selected

def predict_custom_matchup(red_prof, blue_prof, df, model, features):
    red_name = red_prof['name']
    blue_name = blue_prof['name']
    print(f"\n🔮 Analyzing matchup: {red_name} (Red) vs {blue_name} (Blue)...")

    synth = {}
    for f in features:
        if f == 'sig_str_dif' or f == 'R_avg_SIG_STR_landed' or f == 'B_avg_SIG_STR_landed':
            synth['sig_str_dif'] = red_prof['sig_str_landed'] - blue_prof['sig_str_landed']
            synth['R_avg_SIG_STR_landed'] = red_prof['sig_str_landed']
            synth['B_avg_SIG_STR_landed'] = blue_prof['sig_str_landed']
        elif f == 'age_dif' or f == 'R_age' or f == 'B_age':
            synth['age_dif'] = red_prof['age'] - blue_prof['age']
            synth['R_age'] = red_prof['age']
            synth['B_age'] = blue_prof['age']
        elif f == 'reach_dif':
            synth['reach_dif'] = red_prof['reach'] - blue_prof['reach']
        elif f == 'height_dif':
            synth['height_dif'] = red_prof['height'] - blue_prof['height']
        elif f == 'win_streak_dif':
            synth['win_streak_dif'] = red_prof['win_streak'] - blue_prof['win_streak']
        else:
            synth[f] = 0.0

    X_pred = pd.DataFrame([synth])[features].fillna(0)

    win_prob = model.predict_proba(X_pred)[0]
    red_prob = win_prob[1]
    blue_prob = win_prob[0]

    print("=" * 45)
    print(f"🥊 PREDICTION RESULTS:")
    print(f"🔴 Red Corner ({red_name}): {red_prob * 100:.2f}% probability")
    print(f"🔵 Blue Corner ({blue_name}): {blue_prob * 100:.2f}% probability")
    
    winner = red_name if red_prob > blue_prob else blue_name
    print(f"🏆 Predicted Winner: {winner}")
    print("=" * 45)

if __name__ == "__main__":
    print("--- UFC Matchup Predictor Terminal Interface (Global Registry + Patches) ---")
    print("Type 'q' or 'exit' at any time to quit.\n")
    
    csv_path = os.path.join("data", "raw", "ufc-master.csv")
    if os.path.exists(csv_path):
        global_df = pd.read_csv(csv_path)
    else:
        print("❌ Dataset missing. Please check your data directory.")
        exit(1)

    print("⏳ Building global fighter registry...")
    fighter_registry = build_global_fighter_registry(global_df)
    print(f"✅ Loaded {len(fighter_registry)} fighters into memory!")

    model, features = load_model_and_artifacts()
    if model is None:
        exit(1)

    while True:
        red_prof = select_fighter("Enter Red Corner Fighter (partial name ok): ", fighter_registry, global_df)
        if not red_prof:
            break
            
        blue_prof = select_fighter("Enter Blue Corner Fighter (partial name ok): ", fighter_registry, global_df)
        if not blue_prof:
            break
            
        predict_custom_matchup(red_prof, blue_prof, global_df, model, features)
        print("\n" + "-"*45 + "\n")