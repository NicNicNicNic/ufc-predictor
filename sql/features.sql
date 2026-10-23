-- Phase 3: pre-fight features, built only from information available BEFORE each fight.
-- Re-runnable: views are dropped and recreated.
-- Requires SQLite 3.25+ (window functions), which any recent Python includes.

DROP VIEW IF EXISTS fight_ml_features;
DROP VIEW IF EXISTS fight_odds;
DROP VIEW IF EXISTS prefight_features;
DROP VIEW IF EXISTS fighter_form;
DROP VIEW IF EXISTS fighter_fight_log;

-- 1. One row per fighter per fight, with a simple won flag (NULL for draws / no contests).
CREATE VIEW fighter_fight_log AS
SELECT f.fight_id, e.event_date, f.fighter_1_id AS fighter_id,
       CASE WHEN f.winner_id IS NULL THEN NULL
            WHEN f.winner_id = f.fighter_1_id THEN 1 ELSE 0 END AS won
FROM fights f JOIN events e ON e.event_id = f.event_id
UNION ALL
SELECT f.fight_id, e.event_date, f.fighter_2_id AS fighter_id,
       CASE WHEN f.winner_id IS NULL THEN NULL
            WHEN f.winner_id = f.fighter_2_id THEN 1 ELSE 0 END AS won
FROM fights f JOIN events e ON e.event_id = f.event_id;

-- 2. Window functions. Every frame ends at "1 PRECEDING", so the current fight is
--    never included. That is what keeps results from leaking into their own features.
CREATE VIEW fighter_form AS
SELECT
    fight_id,
    fighter_id,
    event_date,
    -- days since the fighter's previous fight (NULL for a first fight in this data)
    JULIANDAY(event_date) - JULIANDAY(
        LAG(event_date) OVER (PARTITION BY fighter_id ORDER BY event_date, fight_id)
    ) AS days_since_last_fight,
    -- fights before this one (counts only fights in this dataset, which starts in 2010)
    COUNT(*) OVER (
        PARTITION BY fighter_id ORDER BY event_date, fight_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_ufc_fights,
    -- recent form: results of the previous 3 fights
    SUM(won) OVER (
        PARTITION BY fighter_id ORDER BY event_date, fight_id
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS wins_last3,
    COUNT(won) OVER (
        PARTITION BY fighter_id ORDER BY event_date, fight_id
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS fights_last3
FROM fighter_fight_log;

-- 3. Per-fighter, per-fight features (the dataset's precomputed pre-fight stats + form).
CREATE VIEW prefight_features AS
SELECT
    s.fight_id,
    s.fighter_id,
    s.height_cm,
    s.reach_cm,
    s.age,
    s.wins,
    s.losses,
    s.wins * 1.0 / NULLIF(s.wins + s.losses + s.draws, 0) AS win_rate,
    s.total_rounds_fought,
    s.total_title_bouts,
    s.current_win_streak,
    s.current_lose_streak,
    s.longest_win_streak,
    s.win_by_ko,
    s.win_by_sub,
    s.avg_sig_str_landed,
    s.avg_sig_str_pct,
    s.avg_td_landed,
    s.avg_td_pct,
    s.avg_sub_att,
    ff.days_since_last_fight,
    ff.prior_ufc_fights,
    ff.wins_last3 * 1.0 / NULLIF(ff.fights_last3, 0) AS recent_win_rate
FROM fight_prefight_stats s
JOIN fighter_form ff ON ff.fight_id = s.fight_id AND ff.fighter_id = s.fighter_id;

-- 4. Betting odds, kept OUT of the feature view on purpose. Use them as a benchmark
--    ("does my model beat the market?"), not as inputs.
CREATE VIEW fight_odds AS
SELECT f.fight_id, s1.odds AS fighter_1_odds, s2.odds AS fighter_2_odds
FROM fights f
JOIN fight_prefight_stats s1 ON s1.fight_id = f.fight_id AND s1.fighter_id = f.fighter_1_id
JOIN fight_prefight_stats s2 ON s2.fight_id = f.fight_id AND s2.fighter_id = f.fighter_2_id;

-- 5. The training view: fighter 1 minus fighter 2 for every feature, plus the target.
--    Fights with no winner are excluded so the target is always 0 or 1.
CREATE VIEW fight_ml_features AS
SELECT
    f.fight_id,
    e.event_date,
    f.fighter_1_id,
    f.fighter_2_id,
    f.weight_class,
    f.gender,
    f.title_bout,
    f.no_of_rounds,
    -- physical
    p1.height_cm - p2.height_cm                         AS height_diff,
    p1.reach_cm  - p2.reach_cm                          AS reach_diff,
    p1.age       - p2.age                               AS age_diff,
    -- record and experience
    p1.wins   - p2.wins                                 AS wins_diff,
    p1.losses - p2.losses                               AS losses_diff,
    p1.win_rate - p2.win_rate                           AS win_rate_diff,
    p1.total_rounds_fought - p2.total_rounds_fought     AS total_rounds_diff,
    p1.total_title_bouts   - p2.total_title_bouts       AS title_bouts_diff,
    p1.prior_ufc_fights    - p2.prior_ufc_fights        AS prior_ufc_fights_diff,
    -- streaks and momentum
    p1.current_win_streak  - p2.current_win_streak      AS win_streak_diff,
    p1.current_lose_streak - p2.current_lose_streak     AS lose_streak_diff,
    p1.longest_win_streak  - p2.longest_win_streak      AS longest_win_streak_diff,
    p1.recent_win_rate     - p2.recent_win_rate         AS recent_win_rate_diff,
    -- finishing ability
    p1.win_by_ko  - p2.win_by_ko                        AS ko_wins_diff,
    p1.win_by_sub - p2.win_by_sub                       AS sub_wins_diff,
    -- striking
    p1.avg_sig_str_landed - p2.avg_sig_str_landed       AS sig_str_landed_diff,
    p1.avg_sig_str_pct    - p2.avg_sig_str_pct          AS sig_str_acc_diff,
    -- grappling
    p1.avg_td_landed - p2.avg_td_landed                 AS td_landed_diff,
    p1.avg_td_pct    - p2.avg_td_pct                    AS td_acc_diff,
    p1.avg_sub_att   - p2.avg_sub_att                   AS sub_att_diff,
    -- inactivity
    p1.days_since_last_fight - p2.days_since_last_fight AS days_since_last_fight_diff,
    -- target
    CASE WHEN f.winner_id = f.fighter_1_id THEN 1 ELSE 0 END AS fighter_1_wins
FROM fights f
JOIN events e ON e.event_id = f.event_id
JOIN prefight_features p1 ON p1.fight_id = f.fight_id AND p1.fighter_id = f.fighter_1_id
JOIN prefight_features p2 ON p2.fight_id = f.fight_id AND p2.fighter_id = f.fighter_2_id
WHERE f.winner_id IS NOT NULL;