import os
import pandas as pd
import numpy as np

def engineer_advanced_features():
    print("--- Engineering Comprehensive Features (Including Custom Elo System) ---")
    csv_path = os.path.join("data", "raw", "ufc-master.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Master dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Sort chronologically to track fighter progression correctly
        df = df.sort_values('date').reset_index(drop=True)
    else:
        print("❌ 'date' column missing from dataset.")
        return

    # Trackers for chronological progression
    fighter_history = {}
    fighter_elos = {} # Default baseline Elo for debut/unknown fighters = 1500
    
    r_layoffs, b_layoffs = [], []
    r_last3_sig, b_last3_sig = [], []
    r_elo_list, b_elo_list = [], []

    for idx, row in df.iterrows():
        f_date = row['date']
        r_fighter = str(row.get('R_fighter', '')).strip().lower()
        b_fighter = str(row.get('B_fighter', '')).strip().lower()
        
        r_sig = row.get('R_avg_SIG_STR_landed', 0.0)
        b_sig = row.get('B_avg_SIG_STR_landed', 0.0)

        # --- 1. Fetch Pre-Fight Elo Ratings ---
        r_current_elo = fighter_elos.get(r_fighter, 1500.0)
        b_current_elo = fighter_elos.get(b_fighter, 1500.0)
        
        r_elo_list.append(r_current_elo)
        b_elo_list.append(b_current_elo)

        # --- 2. Temporal Layoffs & Rolling Striking Averages ---
        if r_fighter in fighter_history and fighter_history[r_fighter]:
            prev_fight = fighter_history[r_fighter][-1]
            r_layoffs.append((f_date - prev_fight['date']).days)
            
            past_sigs = [f['sig_str'] for f in fighter_history[r_fighter][-3:]]
            r_last3_sig.append(sum(past_sigs) / len(past_sigs) if past_sigs else r_sig)
        else:
            r_layoffs.append(365)
            r_last3_sig.append(r_sig)

        if b_fighter in fighter_history and fighter_history[b_fighter]:
            prev_fight = fighter_history[b_fighter][-1]
            b_layoffs.append((f_date - prev_fight['date']).days)
            
            past_sigs = [f['sig_str'] for f in fighter_history[b_fighter][-3:]]
            b_last3_sig.append(sum(past_sigs) / len(past_sigs) if past_sigs else b_sig)
        else:
            b_layoffs.append(365)
            b_last3_sig.append(b_sig)

        # --- 3. Determine Match Outcome & Update Elo Post-Fight ---
        # Assuming 'Winner' column specifies 'Red' or 'Blue' (adjust if your column name differs)
        winner = str(row.get('Winner', '')).strip().lower()
        method = str(row.get('finish_method', '')).strip().lower() # e.g., KO/TKO, SUB, DEC

        # Method multiplier: Finishes scale rating adjustments higher than decisions
        k_base = 32.0
        multiplier = 1.2 if ('ko' in method or 'sub' in method) else 1.0

        if 'red' in winner:
            actual_r = 1.0
        elif 'blue' in winner:
            actual_r = 0.0
        else:
            actual_r = 0.5  # Draw or No Contest

        actual_b = 1.0 - actual_r

        # Calculate expected scores
        exp_r = 1.0 / (1.0 + 10 ** ((b_current_elo - r_current_elo) / 400.0))
        exp_b = 1.0 - exp_r

        # Update post-fight Elo
        new_r_elo = r_current_elo + (k_base * multiplier) * (actual_r - exp_r)
        new_r_s_elo = b_current_elo + (k_base * multiplier) * (actual_b - exp_b)

        fighter_elos[r_fighter] = new_r_elo
        fighter_elos[b_fighter] = new_r_s_elo

        # Update historical trackers
        if r_fighter:
            if r_fighter not in fighter_history:
                fighter_history[r_fighter] = []
            fighter_history[r_fighter].append({'date': f_date, 'sig_str': r_sig})
            
        if b_fighter:
            if b_fighter not in fighter_history:
                fighter_history[b_fighter] = []
            fighter_history[b_fighter].append({'date': f_date, 'sig_str': b_sig})

    # --- 4. Assign Features to DataFrame ---
    df['R_elo'] = r_elo_list
    df['B_elo'] = b_elo_list
    df['elo_dif'] = df['R_elo'] - df['B_elo']

    df['R_days_since_last_fight'] = r_layoffs
    df['B_days_since_last_fight'] = b_layoffs
    df['layoff_dif'] = df['R_days_since_last_fight'] - df['B_days_since_last_fight']
    
    df['R_sig_str_last3'] = r_last3_sig
    df['B_sig_str_last3'] = b_last3_sig
    df['sig_str_last3_dif'] = df['R_sig_str_last3'] - df['B_sig_str_last3']

    # --- Defensive & Damage Profiles ---
    sapm_r_cols = [c for c in df.columns if 'absorbed' in c.lower() and 'r_' in c.lower()]
    sapm_b_cols = [c for c in df.columns if 'absorbed' in c.lower() and 'b_' in c.lower()]
    if sapm_r_cols and sapm_b_cols:
        df['SApM_dif'] = df[sapm_r_cols[0]] - df[sapm_b_cols[0]]

    td_def_r_cols = [c for c in df.columns if 'td_def' in c.lower() and 'r_' in c.lower()]
    td_def_b_cols = [c for c in df.columns if 'td_def' in c.lower() and 'b_' in c.lower()]
    if td_def_r_cols and td_def_b_cols:
        df['takedown_def_dif'] = df[td_def_r_cols[0]] - df[td_def_b_cols[0]]

    # --- Grappling Control & Knockdowns ---
    ctrl_r_cols = [c for c in df.columns if ('ctrl' in c.lower() or 'control' in c.lower()) and 'r_' in c.lower()]
    ctrl_b_cols = [c for c in df.columns if ('ctrl' in c.lower() or 'control' in c.lower()) and 'b_' in c.lower()]
    if ctrl_r_cols and ctrl_b_cols:
        df['ctrl_time_dif'] = df[ctrl_r_cols[0]] - df[ctrl_b_cols[0]]

    kd_r_cols = [c for c in df.columns if ('kd' in c.lower() or 'knockdown' in c.lower()) and 'r_' in c.lower()]
    kd_b_cols = [c for c in df.columns if ('kd' in c.lower() or 'knockdown' in c.lower()) and 'b_' in c.lower()]
    if kd_r_cols and kd_b_cols:
        df['knockdown_dif'] = df[kd_r_cols[0]] - df[kd_b_cols[0]]

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print("✅ Successfully engineered all features including dynamic Elo ratings!")

if __name__ == "__main__":
    engineer_advanced_features()