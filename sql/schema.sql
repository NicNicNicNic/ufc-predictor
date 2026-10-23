-- UFC Fight Predictor schema (SQLite)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS fighters (
    fighter_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    height_cm   REAL,
    reach_cm    REAL,
    dob         DATE,               -- not in ufc-master.csv; age is stored per fight instead
    stance      TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name  TEXT NOT NULL,
    event_date  DATE NOT NULL,      -- ISO text, e.g. 2026-03-28
    location    TEXT,
    UNIQUE (event_name, event_date)
);

CREATE TABLE IF NOT EXISTS fights (
    fight_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      INTEGER NOT NULL,
    fighter_1_id  INTEGER NOT NULL, -- red corner
    fighter_2_id  INTEGER NOT NULL, -- blue corner
    winner_id     INTEGER,          -- NULL for draws / no contests
    weight_class  TEXT,
    gender        TEXT,
    title_bout    INTEGER,          -- 1 / 0
    no_of_rounds  INTEGER,          -- scheduled rounds
    win_method    TEXT,
    end_round     INTEGER,
    end_time_sec  INTEGER,          -- total fight time in seconds
    FOREIGN KEY (event_id)     REFERENCES events(event_id),
    FOREIGN KEY (fighter_1_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (fighter_2_id) REFERENCES fighters(fighter_id),
    FOREIGN KEY (winner_id)    REFERENCES fighters(fighter_id),
    CHECK (fighter_1_id <> fighter_2_id),
    UNIQUE (event_id, fighter_1_id, fighter_2_id)
);

-- Round-by-round stats. Stays empty with ufc-master.csv (it only has per-fight data).
CREATE TABLE IF NOT EXISTS fight_rounds (
    round_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id            INTEGER NOT NULL,
    fighter_id          INTEGER NOT NULL,
    round_num           INTEGER NOT NULL,
    sig_strikes_landed  INTEGER,
    sig_strikes_att     INTEGER,
    takedowns_landed    INTEGER,
    takedowns_att       INTEGER,
    ctrl_time_sec       INTEGER,
    FOREIGN KEY (fight_id)   REFERENCES fights(fight_id),
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id),
    UNIQUE (fight_id, fighter_id, round_num)
);

-- One row per fighter per fight: the fighter's record and averages going INTO the fight.
-- These come precomputed from ufc-master.csv.
CREATE TABLE IF NOT EXISTS fight_prefight_stats (
    fight_id                INTEGER NOT NULL,
    fighter_id              INTEGER NOT NULL,
    corner                  TEXT NOT NULL,   -- 'R' or 'B'
    age                     REAL,
    height_cm               REAL,
    reach_cm                REAL,
    weight_lbs              REAL,
    stance                  TEXT,
    wins                    INTEGER,
    losses                  INTEGER,
    draws                   INTEGER,
    current_win_streak      INTEGER,
    current_lose_streak     INTEGER,
    longest_win_streak      INTEGER,
    total_rounds_fought     INTEGER,
    total_title_bouts       INTEGER,
    win_by_ko               INTEGER,
    win_by_sub              INTEGER,
    win_by_dec_unanimous    INTEGER,
    win_by_dec_split        INTEGER,
    win_by_dec_majority     INTEGER,
    win_by_doctor_stoppage  INTEGER,
    avg_sig_str_landed      REAL,
    avg_sig_str_pct         REAL,
    avg_sub_att             REAL,
    avg_td_landed           REAL,
    avg_td_pct              REAL,
    odds                    REAL,
    PRIMARY KEY (fight_id, fighter_id),
    FOREIGN KEY (fight_id)   REFERENCES fights(fight_id),
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
);

CREATE INDEX IF NOT EXISTS idx_events_date      ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_fights_event     ON fights(event_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighter1  ON fights(fighter_1_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighter2  ON fights(fighter_2_id);
CREATE INDEX IF NOT EXISTS idx_rounds_fight     ON fight_rounds(fight_id);
CREATE INDEX IF NOT EXISTS idx_rounds_fighter   ON fight_rounds(fighter_id);
CREATE INDEX IF NOT EXISTS idx_prefight_fighter ON fight_prefight_stats(fighter_id);