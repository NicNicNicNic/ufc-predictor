"""Phase 2 ETL: load data/raw/ufc-master.csv into db/ufc.db.

Run from the project root:   python -m src.etl
Re-running rebuilds every table from scratch.
"""
import pandas as pd

from src.db import RAW_DIR, get_engine, init_schema

CSV_PATH = RAW_DIR / "ufc-master.csv"

# fight_prefight_stats column -> suffix used in the CSV (columns are R_<suffix> and B_<suffix>)
STAT_MAP = {
    "age": "age",
    "height_cm": "Height_cms",
    "reach_cm": "Reach_cms",
    "weight_lbs": "Weight_lbs",
    "stance": "Stance",
    "wins": "wins",
    "losses": "losses",
    "draws": "draw",
    "current_win_streak": "current_win_streak",
    "current_lose_streak": "current_lose_streak",
    "longest_win_streak": "longest_win_streak",
    "total_rounds_fought": "total_rounds_fought",
    "total_title_bouts": "total_title_bouts",
    "win_by_ko": "win_by_KO/TKO",
    "win_by_sub": "win_by_Submission",
    "win_by_dec_unanimous": "win_by_Decision_Unanimous",
    "win_by_dec_split": "win_by_Decision_Split",
    "win_by_dec_majority": "win_by_Decision_Majority",
    "win_by_doctor_stoppage": "win_by_TKO_Doctor_Stoppage",
    "avg_sig_str_landed": "avg_SIG_STR_landed",
    "avg_sig_str_pct": "avg_SIG_STR_pct",
    "avg_sub_att": "avg_SUB_ATT",
    "avg_td_landed": "avg_TD_landed",
    "avg_td_pct": "avg_TD_pct",
    "odds": "odds",
}

SHARED_COLS = [
    "date", "location", "Winner", "title_bout", "weight_class", "gender",
    "no_of_rounds", "finish", "finish_round", "total_fight_time_secs",
]


def check_columns(df):
    needed = list(SHARED_COLS)
    for p in ("R", "B"):
        needed.append(f"{p}_fighter")
        needed += [f"{p}_{suffix}" for suffix in STAT_MAP.values()]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{CSV_PATH.name} is missing expected columns: {missing}")


def clean(df):
    start = len(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "R_fighter", "B_fighter"])
    for c in ("R_fighter", "B_fighter"):
        df[c] = df[c].astype(str).str.strip()
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df = df[df["R_fighter"] != df["B_fighter"]]
    df["event_date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["event_date", "location", "R_fighter", "B_fighter"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"Rows read: {start}, kept after cleaning: {len(df)}")
    return df


def corner_frame(df, prefix, corner):
    out = pd.DataFrame({
        "fight_id": df["fight_id"],
        "name": df[f"{prefix}_fighter"],
        "date": df["date"],
        "corner": corner,
    })
    for col, suffix in STAT_MAP.items():
        out[col] = df[f"{prefix}_{suffix}"]
    return out


def build_tables(df):
    df = df.copy()
    df["fight_id"] = df.index + 1  # ids follow chronological order

    snaps = pd.concat(
        [corner_frame(df, "R", "R"), corner_frame(df, "B", "B")], ignore_index=True
    )

    # fighters: one row per name, physical traits taken from their most recent fight
    fighters = (
        snaps.sort_values("date", kind="stable")
        .drop_duplicates("name", keep="last")[["name", "height_cm", "reach_cm", "stance"]]
        .sort_values("name")
        .reset_index(drop=True)
    )
    fighters.insert(0, "fighter_id", fighters.index + 1)
    fid = dict(zip(fighters["name"], fighters["fighter_id"]))

    # events: the CSV has no event names, so one event per date + location
    events = (
        df[["event_date", "location"]]
        .drop_duplicates()
        .sort_values(["event_date", "location"])
        .reset_index(drop=True)
    )
    events.insert(0, "event_id", events.index + 1)
    events["event_name"] = "UFC event in " + events["location"]
    df = df.merge(events[["event_id", "event_date", "location"]], on=["event_date", "location"], how="left")

    df["fighter_1_id"] = df["R_fighter"].map(fid)  # red corner
    df["fighter_2_id"] = df["B_fighter"].map(fid)  # blue corner
    winner = df["fighter_1_id"].where(
        df["Winner"] == "Red", df["fighter_2_id"].where(df["Winner"] == "Blue")
    )

    fights = pd.DataFrame({
        "fight_id": df["fight_id"],
        "event_id": df["event_id"],
        "fighter_1_id": df["fighter_1_id"],
        "fighter_2_id": df["fighter_2_id"],
        "winner_id": winner.astype("Int64"),
        "weight_class": df["weight_class"],
        "gender": df["gender"],
        "title_bout": df["title_bout"].map({True: 1, False: 0, "True": 1, "False": 0}),
        "no_of_rounds": pd.to_numeric(df["no_of_rounds"], errors="coerce").astype("Int64"),
        "win_method": df["finish"],
        "end_round": pd.to_numeric(df["finish_round"], errors="coerce").round().astype("Int64"),
        "end_time_sec": pd.to_numeric(df["total_fight_time_secs"], errors="coerce").round().astype("Int64"),
    })

    snaps["fighter_id"] = snaps["name"].map(fid)
    snaps = snaps.drop(columns=["name", "date"])

    return fighters, events[["event_id", "event_name", "event_date", "location"]], fights, snaps


def report(engine):
    print("\nRows loaded:")
    for t in ("fighters", "events", "fights", "fight_rounds", "fight_prefight_stats"):
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", engine)["n"][0]
        print(f"  {t:22s} {n}")
    q = pd.read_sql(
        "SELECT MIN(event_date) AS first, MAX(event_date) AS last FROM events", engine
    )
    print(f"Date range: {q['first'][0]} to {q['last'][0]}")
    nw = pd.read_sql("SELECT COUNT(*) AS n FROM fights WHERE winner_id IS NULL", engine)["n"][0]
    print(f"Fights with no winner (draws / no contests): {nw}")


def leakage_check(engine):
    """After a win, a fighter's NEXT fight should show exactly one more win.
    If their win count in the same fight already includes the result, stats are leaky."""
    d = pd.read_sql(
        """
        SELECT s.fighter_id, s.fight_id, s.wins,
               CASE WHEN f.winner_id = s.fighter_id THEN 1 ELSE 0 END AS won
        FROM fight_prefight_stats s
        JOIN fights f ON f.fight_id = s.fight_id
        ORDER BY s.fighter_id, s.fight_id
        """,
        engine,
    )
    d["next_wins"] = d.groupby("fighter_id")["wins"].shift(-1)
    d = d[(d["won"] == 1)].dropna(subset=["wins", "next_wins"])
    if d.empty:
        print("\nLeakage check: not enough data to run.")
        return
    diff = d["next_wins"] - d["wins"]
    print("\nLeakage check (wins at next fight minus wins at this fight, after a win):")
    print(f"  exactly +1 (stats are pre-fight):   {(diff == 1).mean():.1%}")
    print(f"  exactly  0 (would suggest leakage): {(diff == 0).mean():.1%}")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Can't find {CSV_PATH}. Put ufc-master.csv in data/raw/.")
    raw = pd.read_csv(CSV_PATH)
    check_columns(raw)
    df = clean(raw)
    fighters, events, fights, snaps = build_tables(df)

    init_schema(drop_first=True)
    engine = get_engine()
    # parents first so foreign keys are satisfied
    fighters.to_sql("fighters", engine, if_exists="append", index=False)
    events.to_sql("events", engine, if_exists="append", index=False)
    fights.to_sql("fights", engine, if_exists="append", index=False)
    snaps.to_sql("fight_prefight_stats", engine, if_exists="append", index=False)

    report(engine)
    leakage_check(engine)


if __name__ == "__main__":
    main()