import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering, calculating differentials, Bayesian shrinkage 
    for short-record fighters, granular grappling metrics, and advanced tracking.
    """
    df = df.copy()
    
    # Ensure numeric types for core stats, grappling, and advanced features
    stat_cols = [
        'sig_str_landed', 'sig_str_absorbed', 'td_landed', 'sub_att', 
        'ctrl_time_secs', 'td_def_pct', 'td_att', 'sig_str_def_pct', 'layoff_days'
    ]
    
    for col in stat_cols:
        for corner in ['R', 'B']:
            full_col = f'{corner}_{col}'
            if full_col in df.columns:
                df[full_col] = pd.to_numeric(df[full_col], errors='coerce')

    # Weight-class baseline smoothing (Bayesian Priors for low-sample/debut fighters)
    if 'weight_class' in df.columns:
        for col in stat_cols:
            for corner in ['R', 'B']:
                full_col = f'{corner}_{col}'
                if full_col in df.columns:
                    wc_median = df.groupby('weight_class')[full_col].transform('median')
                    df[full_col] = df[full_col].fillna(wc_median)

    # Standard volume & physical differentials
    df['sig_str_dif'] = df.get('R_avg_SIG_STR_landed', 0) - df.get('B_avg_SIG_STR_landed', 0)
    df['age_dif'] = df.get('R_age', 0) - df.get('B_age', 0)
    df['win_streak_dif'] = df.get('R_current_win_streak', 0) - df.get('B_current_win_streak', 0)
    df['loss_dif'] = df.get('R_current_lose_streak', 0) - df.get('B_current_lose_streak', 0)
    df['reach_dif'] = df.get('R_Reach_cms', 0) - df.get('B_Reach_cms', 0)
    df['height_dif'] = df.get('R_Height_cms', 0) - df.get('B_Height_cms', 0)

    # --- GRANULAR GRAPPLING DIFFERENTIALS ---
    df['td_landed_dif'] = df.get('R_avg_td_landed', 0) - df.get('B_avg_td_landed', 0)
    
    if 'R_ctrl_time_secs' in df.columns and 'B_ctrl_time_secs' in df.columns:
        df['ctrl_time_dif'] = df['R_ctrl_time_secs'] - df['B_ctrl_time_secs']
    else:
        df['ctrl_time_dif'] = 0.0

    df['sub_att_dif'] = df.get('R_avg_sub_att', 0) - df.get('B_avg_sub_att', 0)

    if 'R_td_def_pct' in df.columns and 'B_td_def_pct' in df.columns:
        df['td_def_dif'] = df['R_td_def_pct'] - df['B_td_def_pct']
    else:
        df['td_def_dif'] = 0.0

    # --- ADVANCED METRICS: DAMAGE RATIOS & LAYOFFS ---
    r_landed = df.get('R_sig_str_landed', df.get('R_avg_SIG_STR_landed', 1))
    r_absorbed = df.get('R_sig_str_absorbed', 1)
    b_landed = df.get('B_sig_str_landed', df.get('B_avg_SIG_STR_landed', 1))
    b_absorbed = df.get('B_sig_str_absorbed', 1)
    
    df['R_damage_ratio'] = r_landed / (r_absorbed + 1.0)
    df['B_damage_ratio'] = b_landed / (b_absorbed + 1.0)
    df['damage_ratio_dif'] = df['R_damage_ratio'] - df['B_damage_ratio']

    if 'R_layoff_days' in df.columns and 'B_layoff_days' in df.columns:
        df['layoff_dif'] = df['R_layoff_days'] - df['B_layoff_days']
    else:
        df['layoff_dif'] = 0.0

    # Stance interaction feature (Orthodox = 0, Southpaw = 1, Switch = 2)
    if 'R_stance' in df.columns and 'B_stance' in df.columns:
        stance_map = {'orthodox': 0, 'southpaw': 1, 'switch': 2}
        r_stance_val = df['R_stance'].str.strip().str.lower().map(stance_map).fillna(0)
        b_stance_val = df['B_stance'].str.strip().str.lower().map(stance_map).fillna(0)
        df['stance_mismatch'] = (r_stance_val != b_stance_val).astype(int)

    return df