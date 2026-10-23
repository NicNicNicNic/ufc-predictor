import os
import pickle
import pandas as pd

def load_model_and_scaler():
    model_path = os.path.join("models", "ufc_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Run 'python src/train.py' first.")
    with open(model_path, "rb") as f:
        model, scaler, _, _ = pickle.load(f)
    return model, scaler

def get_latest_fighter_stats(df, fighter_name):
    """Searches the dataset for a fighter's most recent stats (Red or Blue corner)."""
    r_fights = df[df['R_fighter'].str.lower() == fighter_name.lower()]
    b_fights = df[df['B_fighter'].str.lower() == fighter_name.lower()]
    
    if r_fights.empty and b_fights.empty:
        return None
        
    if not r_fights.empty and (b_fights.empty or r_fights.index[-1] > b_fights.index[-1]):
        latest = r_fights.iloc[-1]
        return {
            'age': latest.get('R_age', 30),
            'sig_str_landed': latest.get('R_avg_SIG_STR_landed', 0.0),
            'win_streak': latest.get('R_win_streak', 0),
            'losses': latest.get('R_losses', 0),
            'longest_win_streak': latest.get('R_longest_win_streak', 0),
            'height_cms': latest.get('R_height_cms', 175),
            'reach_cms': latest.get('R_reach_cms', 175),
            'avg_sub_att': latest.get('R_avg_SUB_ATT', 0.0),
            'avg_td': latest.get('R_avg_TD_landed', 0.0)
        }
    else:
        latest = b_fights.iloc[-1]
        return {
            'age': latest.get('B_age', 30),
            'sig_str_landed': latest.get('B_avg_SIG_STR_landed', 0.0),
            'win_streak': latest.get('B_win_streak', 0),
            'losses': latest.get('B_losses', 0),
            'longest_win_streak': latest.get('B_longest_win_streak', 0),
            'height_cms': latest.get('B_height_cms', 175),
            'reach_cms': latest.get('B_reach_cms', 175),
            'avg_sub_att': latest.get('B_avg_SUB_ATT', 0.0),
            'avg_td': latest.get('B_avg_TD_landed', 0.0)
        }

def predict_custom_matchup(red_fighter, blue_fighter):
    data_path = os.path.join("data", "raw", "ufc-master.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")
        
    df = pd.read_csv(data_path)
    
    r_stats = get_latest_fighter_stats(df, red_fighter)
    b_stats = get_latest_fighter_stats(df, blue_fighter)
    
    if not r_stats:
        print(f"❌ Error: Fighter '{red_fighter}' not found in database.")
        return
    if not b_stats:
        print(f"❌ Error: Fighter '{blue_fighter}' not found in database.")
        return
        
    diff_vector = {
        'reach_dif': r_stats['reach_cms'] - b_stats['reach_cms'],
        'age_dif': r_stats['age'] - b_stats['age'],
        'height_dif': r_stats['height_cms'] - b_stats['height_cms'],
        'win_streak_dif': r_stats['win_streak'] - b_stats['win_streak'],
        'loss_dif': r_stats['losses'] - b_stats['losses'],
        'longest_win_streak_dif': r_stats['longest_win_streak'] - b_stats['longest_win_streak'],
        'sig_str_dif': r_stats['sig_str_landed'] - b_stats['sig_str_landed'],
        'avg_sub_att_dif': r_stats['avg_sub_att'] - b_stats['avg_sub_att'],
        'avg_td_dif': r_stats['avg_td'] - b_stats['avg_td'],
        'R_age': r_stats['age'],
        'B_age': b_stats['age'],
        'R_avg_SIG_STR_landed': r_stats['sig_str_landed'],
        'B_avg_SIG_STR_landed': b_stats['sig_str_landed']
    }
    
    feature_df = pd.DataFrame([diff_vector])
    model, scaler = load_model_and_scaler()
    scaled_features = scaler.transform(feature_df)
    
    proba = model.predict_proba(scaled_features)[0]
    red_prob = proba[1] * 100
    blue_prob = proba[0] * 100
    
    winner = red_fighter if red_prob > blue_prob else blue_fighter
    confidence = max(red_prob, blue_prob)
    
    print("\n" + "=" * 50)
    print("            MATCHUP PREDICTION RESULT")
    print("=" * 50)
    print(f"🔴 {red_fighter}: {red_prob:.1f}% win probability")
    print(f"🔵 {blue_fighter}: {blue_prob:.1f}% win probability")
    print("-" * 50)
    print(f"🏆 Predicted Winner: {winner} ({confidence:.1f}% confidence)")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    print("\n--- UFC Matchup Predictor Terminal Interface ---")
    print("Type 'q' or 'exit' at any time to quit.\n")
    
    while True:
        red_input = input("Enter Red Corner Fighter: ").strip()
        if red_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting predictor. See ya!")
            break
            
        blue_input = input("Enter Blue Corner Fighter: ").strip()
        if blue_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting predictor. See ya!")
            break
            
        if not red_input or not blue_input:
            print("⚠️ Please enter valid fighter names.\n")
            continue
            
        predict_custom_matchup(red_input, blue_input)