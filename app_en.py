"""
Football Analytics Dashboard v3.1 — English Edition
=====================================================
5-Metric Framework: Process × Critical × Luck
2022 FIFA World Cup — All 64 Matches
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsbombpy import sb

warnings.filterwarnings("ignore")

# =========================================================
# StatsBomb Attribution (required by Open Data User Agreement)
# =========================================================
SB_LOGO_URL = (
    "https://raw.githubusercontent.com/statsbomb/logos/main/"
    "HudlStatsbomb_Python.svg"
)

def render_sb_sidebar():
    """Display StatsBomb credit in sidebar (required)."""
    st.sidebar.markdown("---")
    st.sidebar.image(SB_LOGO_URL, width=160)
    st.sidebar.markdown(
        "**Data provided by StatsBomb**  \n"
        "© StatsBomb Services Ltd.  \n"
        "[statsbomb.com](https://statsbomb.com) · "
        "[Open Data](https://github.com/statsbomb/open-data)  \n"
        "*Non-commercial use only*"
    )

def render_sb_footer():
    """Display StatsBomb credit at page bottom (required)."""
    st.markdown("---")
    col_logo, col_text = st.columns([1, 6])
    with col_logo:
        st.image(SB_LOGO_URL, width=130)
    with col_text:
        st.markdown(
            "**Data source: StatsBomb Open Data**  \n"
            "This application uses data made freely available by StatsBomb Services Ltd. "
            "under the [StatsBomb Open Data User Agreement]"
            "(https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).  \n"
            "Free for non-commercial use only. "
            "© StatsBomb Services Ltd. All rights reserved."
        )


# =========================================================
# Column name mapping  (internal keys stay in Japanese
# so compute_raw_scores() is unchanged; display uses EN)
# =========================================================
COL = {
    "atk":      "①攻撃プロセス",
    "def":      "②守備プロセス",
    "g_threat": "③得点近接",
    "save":     "④失点近接",
    "process":  "総合プロセス(①+②)",
    "critical": "総合クリティカル(③+④)",
    "g_luck":   "⑤得点Luck(決定力)",
    "d_luck":   "⑤守備Luck(死守度)",
    # team summary keys
    "atk_sum":  "攻撃プロセス合計",
    "def_sum":  "守備プロセス合計",
    "proc_sum": "プロセス合計(絶対)",
    "ca_sum":   "攻撃クリティカル",
    "cd_sum":   "守備クリティカル",
    "crit_sum": "クリティカル合計",
}

# Display labels for DataFrames
EN_LABELS = {
    "①攻撃プロセス":          "① Attack Process",
    "②守備プロセス":          "② Defense Process",
    "③得点近接":              "③ Goal Threat",
    "④失点近接":              "④ Save Contrib",
    "総合プロセス(①+②)":     "Process Total (①+②)",
    "総合クリティカル(③+④)":  "Critical Total (③+④)",
}

RADAR_COLS = [
    "①攻撃プロセス", "②守備プロセス", "③得点近接",
    "④失点近接", "総合クリティカル(③+④)",
]
RADAR_LABELS = [
    "① Attack\nProcess", "② Defense\nProcess", "③ Goal\nThreat",
    "④ Save\nContrib", "Critical\nTotal",
]

# =========================================================
# xT Grid (12×8)
# =========================================================
XT_GRID = np.array([
    [0.00,0.00,0.00,0.00,0.01,0.02,0.03,0.04],
    [0.00,0.00,0.00,0.01,0.01,0.02,0.03,0.04],
    [0.00,0.00,0.01,0.01,0.02,0.03,0.04,0.06],
    [0.00,0.01,0.01,0.02,0.03,0.04,0.06,0.09],
    [0.01,0.01,0.02,0.03,0.04,0.06,0.09,0.14],
    [0.01,0.02,0.03,0.04,0.06,0.08,0.12,0.18],
    [0.02,0.03,0.04,0.06,0.08,0.12,0.18,0.25],
    [0.03,0.04,0.06,0.08,0.10,0.15,0.22,0.35],
    [0.04,0.05,0.07,0.10,0.13,0.20,0.30,0.45],
    [0.05,0.07,0.09,0.13,0.18,0.28,0.40,0.60],
    [0.07,0.09,0.12,0.18,0.25,0.38,0.55,0.75],
    [0.10,0.13,0.18,0.25,0.35,0.50,0.70,0.90],
])

def get_xt(x: float, y: float) -> float:
    return float(XT_GRID[min(int(x/10),11)][min(int(y/10),7)])


# =========================================================
# Core Analysis (unchanged logic, same as JP version)
# =========================================================
def compute_raw_scores(match_id: int):
    events = sb.events(match_id=match_id)
    events = events.dropna(subset=["player"]).copy()
    teams  = events["team"].dropna().unique()
    if len(teams) < 2:
        return None

    events["time_seconds"] = (
        events["minute"]*60 + events["second"]
        + (events["period"]-1)*45*60
    )
    events = events.sort_values("time_seconds").reset_index(drop=True)

    players = events[["player","team"]].drop_duplicates().set_index("player")
    scores  = pd.DataFrame(index=players.index)
    scores["team"]     = players["team"]
    scores["match_id"] = match_id

    pa  = dict.fromkeys(scores.index, 0.0)
    pd_ = dict.fromkeys(scores.index, 0.0)
    ca  = dict.fromkeys(scores.index, 0.0)
    cd  = dict.fromkeys(scores.index, 0.0)

    passes = events[events["type"]=="Pass"].copy()
    shots  = events[events["type"]=="Shot"].copy()

    # ① Attack Process
    for _, ev in passes.iterrows():
        try:
            sx,sy = ev["location"][0], ev["location"][1]
            ex,ey = ev["pass_end_location"][0], ev["pass_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue
        passer   = ev["player"]
        receiver = ev.get("pass_recipient")
        xt_gain  = get_xt(ex,ey) - get_xt(sx,sy)
        if pd.isna(ev.get("pass_outcome")):
            if xt_gain > 0:
                pa[passer] += xt_gain * 2.0
                if receiver in pa: pa[receiver] += xt_gain * 8.0
            if sx < 80 <= ex:
                pa[passer] += 1.0
                if receiver in pa: pa[receiver] += 0.5
            if ex >= 102 and 18 <= ey <= 62:
                pa[passer] += 1.5
                if receiver in pa: pa[receiver] += 1.5
            if ev.get("under_pressure") is True and xt_gain > 0:
                pa[passer] += 0.5
        else:
            pa[passer] -= 3.0 if sx < 40 else (1.0 if sx < 60 else 0.3)

    for _, ev in events[events["type"]=="Carry"].copy().iterrows():
        try:
            sx,sy = ev["location"][0], ev["location"][1]
            ex,ey = ev["carry_end_location"][0], ev["carry_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue
        xt_gain = get_xt(ex,ey) - get_xt(sx,sy)
        if xt_gain > 0: pa[ev["player"]] += xt_gain * 15.0

    dribbles = events[events["type"]=="Dribble"]
    if "dribble_outcome" in dribbles.columns:
        for _, ev in dribbles[dribbles["dribble_outcome"]=="Complete"].iterrows():
            if ev["player"] in pa:
                try:
                    pa[ev["player"]] += 1.5 if ev["location"][0] >= 80 else 0.5
                except (TypeError, IndexError):
                    pa[ev["player"]] += 0.5

    for _, ev in events[events["type"]=="Foul Won"].iterrows():
        if ev["player"] in pa:
            try:
                pa[ev["player"]] += 0.5 if ev["location"][0] >= 80 else 0.2
            except (TypeError, IndexError):
                pa[ev["player"]] += 0.2

    # ② Defense Process
    def def_weight(loc, atype="default"):
        try:    x = float(loc[0])
        except: x = 60.0
        if atype == "interception":
            return 3.0 if x>=80 else (2.0 if x>=60 else (1.5 if x>=40 else 1.2))
        elif atype == "duel":
            return 2.5 if x>=80 else (1.5 if x>=60 else (0.8 if x>=40 else 0.3))
        elif atype == "clearance":
            return 2.0 if x<20 else (0.5 if x<40 else 0.8)
        else:
            return 2.0 if x>=80 else (1.0 if x>=40 else 0.6)

    for _, ev in events[events["type"]=="Interception"].iterrows():
        if ev["player"] in pd_: pd_[ev["player"]] += def_weight(ev.get("location"),"interception")
    for _, ev in events[events["type"]=="Ball Recovery"].iterrows():
        if ev["player"] in pd_: pd_[ev["player"]] += def_weight(ev.get("location"),"recovery")
    duels = events[events["type"]=="Duel"]
    if "duel_outcome" in duels.columns:
        for _, ev in duels[duels["duel_outcome"].isin(["Won","Success","Tackle"])].iterrows():
            if ev["player"] in pd_: pd_[ev["player"]] += def_weight(ev.get("location"),"duel")
    for _, ev in events[events["type"]=="Clearance"].iterrows():
        if ev["player"] in pd_: pd_[ev["player"]] += def_weight(ev.get("location"),"clearance")
    for _, ev in events[events["type"]=="Pressure"].iterrows():
        if ev["player"] in pd_:
            try:
                if float(ev["location"][0]) >= 80: pd_[ev["player"]] += 0.5
            except: pass

    # ③ Goal Threat
    for idx in shots.index:
        shot   = events.loc[idx]
        xg     = shot.get("shot_statsbomb_xg", 0)
        if pd.isna(xg): xg = 0.0
        xg_adj = xg * (1.2 if shot.get("shot_body_part") in ["Head","No Touch"] else 1.0)
        shooter = shot["player"]
        if shooter in ca: ca[shooter] += xg_adj * 0.50
        contributors, pos = [], events.index.get_loc(idx)
        for step in range(1, 20):
            prev_pos = pos - step
            if prev_pos < 0: break
            prev = events.iloc[prev_pos]
            if prev["team"] != shot["team"]: break
            pl = prev["player"]
            if pl != shooter and pl not in contributors: contributors.append(pl)
            if len(contributors) >= 2: break
        for j, c in enumerate(contributors[:2]):
            if c in ca: ca[c] += xg_adj * ([0.30,0.20][j])

    # ④ Save Contribution
    loss_events = events[events["type"].isin(["Miscontrol","Dispossessed"])]
    for _, loss in loss_events.iterrows():
        t = loss["time_seconds"]
        fs = shots[(shots["team"]!=loss["team"]) &
                   (shots["time_seconds"]>=t) & (shots["time_seconds"]<=t+6)]
        if not fs.empty:
            penalty = fs["shot_statsbomb_xg"].fillna(0).sum()
            if loss["player"] in cd: cd[loss["player"]] -= penalty * 1.2
    for _, ev in events[events["type"]=="Goal Keeper"].iterrows():
        if ev.get("goalkeeper_type") in ["Save","Shot Saved"] and ev["player"] in cd:
            sn = shots[abs(shots["time_seconds"]-ev["time_seconds"])<2]
            cd[ev["player"]] += max(sn["shot_statsbomb_xg"].fillna(0.1).sum(), 0.1)
    for _, ev in events[events["type"]=="Block"].iterrows():
        if ev["player"] in cd:
            sn = shots[abs(shots["time_seconds"]-ev["time_seconds"])<2]
            cd[ev["player"]] += max(sn["shot_statsbomb_xg"].fillna(0.05).sum()*0.8, 0.05)

    # Assemble
    xg_team   = shots.groupby("team")["shot_statsbomb_xg"].sum()
    team_goals = {}
    for _, s in shots.iterrows():
        if s.get("shot_outcome") == "Goal":
            team_goals[s["team"]] = team_goals.get(s["team"],0)+1

    scores["①攻撃プロセス_raw"]     = pd.Series(pa)
    scores["②守備プロセス_raw"]     = pd.Series(pd_)
    scores["③得点近接"]             = pd.Series(ca)
    scores["④失点近接"]             = pd.Series(cd)
    scores["総合クリティカル(③+④)"] = scores["③得点近接"] + scores["④失点近接"]

    team_summaries = {}
    for t in events["team"].dropna().unique():
        opp_list = [x for x in events["team"].dropna().unique() if x != t]
        opp  = opp_list[0] if opp_list else t
        xg_t = float(xg_team.get(t,0))
        xg_o = float(xg_team.get(opp,0))
        g_t  = team_goals.get(t,0)
        g_o  = team_goals.get(opp,0)
        tm   = scores[scores["team"]==t]
        team_summaries[t] = {
            "Goals": g_t, "Goals Against": g_o,
            "xG": xg_t, "xG Against": xg_o,
            "攻撃クリティカル": float(tm["③得点近接"].sum()),
            "守備クリティカル": float(tm["④失点近接"].sum()),
            "クリティカル合計": float(tm["総合クリティカル(③+④)"].sum()),
            "⑤得点Luck(決定力)": g_t - xg_t,
            "⑤守備Luck(死守度)": xg_o - g_o,
        }
    return scores.reset_index(), team_summaries


# =========================================================
# Full-Tournament Pool (absolute normalisation)
# =========================================================
@st.cache_data(show_spinner="Analysing all 64 matches... (first run only, ~2 min)")
def build_pool():
    matches    = sb.matches(competition_id=43, season_id=106)
    all_scores = []
    match_meta = {}
    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        try:
            result = compute_raw_scores(mid)
            if result is None: continue
            df_s, team_sum = result
            all_scores.append(df_s)
            match_meta[mid] = {
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "match_date": row["match_date"],
                "team_summaries": team_sum,
            }
        except Exception:
            pass

    if not all_scores:
        return None, None

    pool = pd.concat(all_scores, ignore_index=True)
    for raw, norm in [("①攻撃プロセス_raw","①攻撃プロセス"),
                      ("②守備プロセス_raw","②守備プロセス")]:
        mu, sd = pool[raw].mean(), pool[raw].std()
        pool[norm] = (pool[raw]-mu) / (sd if sd>0 else 1.0)
    pool["総合プロセス(①+②)"] = pool["①攻撃プロセス"] + pool["②守備プロセス"]
    return pool, match_meta


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Football Analytics Dashboard",
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ 5-Metric Football Analytics Dashboard")
st.markdown(
    "**① Attack Process &nbsp;② Defense Process &nbsp;"
    "③ Goal Threat &nbsp;④ Save Contribution &nbsp;⑤ Luck Score**  \n"
    "2022 FIFA World Cup — All 64 Matches &nbsp;|&nbsp; "
    "Absolute scoring: every player rated on the same tournament-wide scale"
)

# =========================================================
# Plain-English metric guide
# =========================================================
with st.expander("📖 What do the 5 metrics mean? (Start here)", expanded=False):
    st.markdown("""
    ---
    ### ① Attack Process
    > **"How much did this player move the ball toward danger?"**

    Every pass and dribble is scored by how much it increased the probability of a goal
    — based on where the ball started and ended (xT model).  
    **Receivers and dribblers are rewarded more than passers**, so forwards and wingers
    naturally score higher than centre-backs just playing safe sideways passes.

    ---
    ### ② Defense Process
    > **"How aggressively did this player disrupt the opponent?"**

    High-press actions (winning the ball in the opponent's half) earn the most points.
    Routine clearances in a defender's own box earn very little — the metric rewards
    *proactive* defending, not just staying put.

    ---
    ### ③ Goal Threat
    > **"How involved was this player in actual scoring chances?"**

    Each shot is weighted by its xG (expected goals). The credit is split:
    **50 %** to the shooter, **30 %** to the player who played the final pass,
    **20 %** to the player before that.  
    A big number here means the player was regularly in the thick of dangerous attacks.

    ---
    ### ④ Save Contribution
    > **"How much did this player directly prevent goals?"**

    Saves and blocks are rewarded in proportion to how dangerous the shot was
    (xG-weighted). Turnovers that led to an opponent shot within 6 seconds are penalised.  
    A positive number = net contribution to keeping the score down.  
    A negative number = the player's mistakes created more danger than they stopped.

    ---
    ### ⑤ Luck Score
    > **"How much did chance affect the scoreline?"**

    - **Goal Luck** = Actual goals − xG &nbsp;→ positive means the team scored *more* than the
      chances warranted (clinical finishing or fortunate bounces).  
    - **Defense Luck** = xG Against − Goals conceded &nbsp;→ positive means the team conceded
      *fewer* than expected (great keeping or good fortune).

    This separates **real quality** from **randomness**.

    ---
    ### Reading the numbers (① & ②)
    | Score | Meaning |
    |-------|---------|
    | **+2.0 or above** | World-class for this tournament (top ~2–3 %) |
    | **+1.0 to +2.0** | Excellent (top 16 %) |
    | **~0.0** | Tournament average |
    | **−1.0 or below** | Well below average |

    ③ and ④ are raw xG-based values (not z-scored) — compare them within a match context.
    """)

# =========================================================
# Load pool
# =========================================================
with st.spinner("Loading data..."):
    pool_df, match_meta = build_pool()

if pool_df is None:
    st.error("Failed to load match data.")
    st.stop()

render_sb_sidebar()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Page", ["🔍 Match Analysis", "🏆 Tournament Rankings"])
st.sidebar.markdown("---")
st.sidebar.header("🔍 Match Filter")
search_q = st.sidebar.text_input("Filter by team name (e.g. Japan)", "").lower()

labels, label_to_mid = [], {}
for mid, m in match_meta.items():
    label = (f"{m['match_date']} | "
             f"{m['home_team']} {m['home_score']} – "
             f"{m['away_score']} {m['away_team']}")
    if not search_q or search_q in label.lower():
        labels.append(label)
        label_to_mid[label] = mid

if not labels:
    st.sidebar.warning("No matches found.")
    st.stop()

selected_label = st.sidebar.selectbox("Select match", sorted(labels))
match_id = label_to_mid[selected_label]
meta     = match_meta[match_id]
home_t, away_t = meta["home_team"], meta["away_team"]

df_match = pool_df[pool_df["match_id"]==match_id].copy()
team_sum = meta["team_summaries"]

for t in [home_t, away_t]:
    tm = df_match[df_match["team"]==t]
    team_sum[t]["プロセス合計(絶対)"] = float(tm["総合プロセス(①+②)"].sum())
    team_sum[t]["攻撃プロセス合計"]   = float(tm["①攻撃プロセス"].sum())
    team_sum[t]["守備プロセス合計"]   = float(tm["②守備プロセス"].sum())


# =========================================================
# Helpers
# =========================================================
def rename_cols(df):
    """Rename internal Japanese column keys to English display labels."""
    return df.rename(columns=EN_LABELS)

def fmt_df(df, sort_col):
    num_cols = [c for c in df.columns if c not in ("player","team","rank")]
    return df.sort_values(sort_col, ascending=False).style \
        .background_gradient(subset=[sort_col], cmap="RdYlGn") \
        .format({c: "{:+.3f}" for c in num_cols})


# =========================================================
# PAGE A — Match Analysis
# =========================================================
if page == "🔍 Match Analysis":

    st.subheader(f"📊  {home_t}  vs  {away_t}")
    st.caption(f"Score: {meta['home_score']} – {meta['away_score']}  |  {meta['match_date']}")

    col_h, col_sep, col_a = st.columns([5, 1, 5])

    def render_team_card(col, team, data):
        with col:
            g_for = data["Goals"]
            g_ag  = data["Goals Against"]
            result = "🏆" if g_for > g_ag else ("🤝" if g_for == g_ag else "❌")
            st.markdown(f"### {result}  {team}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Goals", g_for)
            c2.metric("xG (expected)", f"{data['xG']:.2f}",
                      delta=f"Luck {data['⑤得点Luck(決定力)']:+.2f}")
            c3.metric("xG Against", f"{data['xG Against']:.2f}",
                      delta=f"Luck {data['⑤守備Luck(死守度)']:+.2f}")

            lk_a = data["⑤得点Luck(決定力)"]
            lk_d = data["⑤守備Luck(死守度)"]
            st.markdown(
                f"🎯 **Goal Luck:** `{lk_a:+.2f}` "
                + ("✨ Scored above expectation" if lk_a > 0.3
                   else "⚡ Around expectation" if lk_a > -0.3
                   else "😬 Missed chances")
            )
            st.markdown(
                f"🛡️ **Defense Luck:** `{lk_d:+.2f}` "
                + ("🧱 Conceded below expectation" if lk_d > 0.3
                   else "⚡ Around expectation" if lk_d > -0.3
                   else "💥 Conceded above expectation")
            )
            proc = data.get("プロセス合計(絶対)", 0)
            crit = data["クリティカル合計"]
            st.info(
                f"Game Control (Process total): **{proc:+.2f}**  \n"
                f"Decisive Moments (Critical total): **{crit:+.2f}**"
            )

    render_team_card(col_h, home_t, team_sum[home_t])
    with col_sep:
        st.markdown(
            "<div style='text-align:center;padding-top:80px;font-size:24px'>VS</div>",
            unsafe_allow_html=True
        )
    render_team_card(col_a, away_t, team_sum[away_t])

    # --- Team comparison bar charts ---
    st.markdown("---")
    st.subheader("📈 Team Metric Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    w = 0.35

    proc_labels = ["Attack\nProcess", "Defense\nProcess", "Process\nTotal"]
    proc_keys   = ["攻撃プロセス合計", "守備プロセス合計", "プロセス合計(絶対)"]
    xp = np.arange(len(proc_labels))
    axes[0].bar(xp-w/2, [team_sum[home_t].get(k,0) for k in proc_keys],
                w, label=home_t, color="#e74c3c", alpha=0.85)
    axes[0].bar(xp+w/2, [team_sum[away_t].get(k,0) for k in proc_keys],
                w, label=away_t, color="#3498db", alpha=0.85)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(proc_labels, fontsize=9)
    axes[0].legend()
    axes[0].set_title("Process Metrics (Absolute — WC-wide z-score)")

    crit_labels = ["Goal\nThreat", "Save\nContrib", "Critical\nTotal",
                   "Goal\nLuck", "Defense\nLuck"]
    crit_keys   = ["攻撃クリティカル", "守備クリティカル", "クリティカル合計",
                   "⑤得点Luck(決定力)", "⑤守備Luck(死守度)"]
    xc = np.arange(len(crit_labels))
    axes[1].bar(xc-w/2, [team_sum[home_t].get(k,0) for k in crit_keys],
                w, label=home_t, color="#e74c3c", alpha=0.85)
    axes[1].bar(xc+w/2, [team_sum[away_t].get(k,0) for k in crit_keys],
                w, label=away_t, color="#3498db", alpha=0.85)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xticks(xc)
    axes[1].set_xticklabels(crit_labels, fontsize=9)
    axes[1].legend()
    axes[1].set_title("Critical & Luck Metrics")

    plt.tight_layout()
    st.pyplot(fig)

    # --- Player stats ---
    st.markdown("---")
    st.subheader("🏃 Player Stats (WC-wide Absolute Rating)")

    target_team = st.radio("Select team", [home_t, away_t], horizontal=True)
    df_team = df_match[df_match["team"]==target_team].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "⭐ All Metrics", "🟢 ① Attack Process",
        "🔵 ② Defense Process", "🔴 ③④ Decisive Moments"
    ])

    all_cols_jp = [
        "player", "総合プロセス(①+②)", "①攻撃プロセス", "②守備プロセス",
        "総合クリティカル(③+④)", "③得点近接", "④失点近接"
    ]
    all_cols_en = {
        "player": "Player",
        **EN_LABELS
    }

    def show_table(df, sort_jp, cmap="RdYlGn", extra_cmaps=None):
        df_en = rename_cols(df.copy())
        sort_en = EN_LABELS.get(sort_jp, sort_jp)
        num_cols = [c for c in df_en.columns if c != "Player"]
        styled = df_en.sort_values(sort_en, ascending=False).style \
            .background_gradient(subset=[sort_en], cmap=cmap) \
            .format({c: "{:+.3f}" for c in num_cols})
        if extra_cmaps:
            for col_en, cm in extra_cmaps.items():
                styled = styled.background_gradient(subset=[col_en], cmap=cm)
        st.dataframe(styled, use_container_width=True)

    with tab1:
        st.caption(
            "Process scores are z-scored against all WC players (mean = 0). "
            "+1.0 = top 16 % of all players in the tournament."
        )
        show_table(df_team[all_cols_jp], "総合プロセス(①+②)")

    with tab2:
        st.caption(
            "Rewards carrying into dangerous areas and receiving progressive passes. "
            "Centre-backs playing safe sideways passes score low by design."
        )
        show_table(df_team[["player","①攻撃プロセス"]], "①攻撃プロセス", cmap="YlGn")

    with tab3:
        st.caption(
            "High-press actions (x ≥ 80 m) earn the highest multiplier. "
            "Routine defensive clearances in a defender's own box score very little."
        )
        show_table(df_team[["player","②守備プロセス"]], "②守備プロセス", cmap="Blues")

    with tab4:
        st.caption(
            "③ Goal Threat: higher is better — player was involved in dangerous attacks.  \n"
            "④ Save Contribution: positive = saved/blocked danger; "
            "negative = turnovers that led to opponent shots."
        )
        df_c = rename_cols(
            df_team[["player","③得点近接","④失点近接","総合クリティカル(③+④)"]].copy()
        ).sort_values("Critical Total (③+④)", ascending=False)
        st.dataframe(
            df_c.style
            .background_gradient(subset=["③ Goal Threat"],  cmap="Oranges")
            .background_gradient(subset=["④ Save Contrib"],  cmap="RdYlGn")
            .format({c: "{:+.3f}" for c in df_c.columns if c != "Player"}),
            use_container_width=True
        )

    # --- Radar chart ---
    st.markdown("---")
    st.subheader("🕸️ Player Radar Chart")
    st.caption("Outer = higher WC-wide percentile rank across all players in the tournament.")

    all_p = sorted(df_team["player"].tolist())
    sel_p = st.multiselect(
        "Select players to compare (2–5 recommended)", all_p,
        default=all_p[:3] if len(all_p) >= 3 else all_p
    )

    if sel_p:
        df_r    = df_team[df_team["player"].isin(sel_p)].set_index("player")
        df_norm = df_r[RADAR_COLS].copy()
        for col in RADAR_COLS:
            pool_vals = pool_df[col].dropna()
            df_norm[col] = df_norm[col].apply(
                lambda v: float((pool_vals <= v).mean())
            )

        n      = len(RADAR_LABELS)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig_r, ax_r = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(sel_p)))

        for player, color in zip(sel_p, colors):
            if player not in df_norm.index: continue
            vals = df_norm.loc[player, RADAR_COLS].tolist() + [df_norm.loc[player, RADAR_COLS[0]]]
            ax_r.plot(angles, vals, "o-", lw=2, label=player, color=color)
            ax_r.fill(angles, vals, alpha=0.15, color=color)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(RADAR_LABELS, size=8)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75])
        ax_r.set_yticklabels(["25th %ile", "50th %ile", "75th %ile"], size=7)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
        ax_r.set_title(f"{target_team} — WC Percentile Radar", size=11, pad=20)
        plt.tight_layout()
        st.pyplot(fig_r)

        st.markdown(
            "**Radar axes:** &nbsp;"
            "`① Attack Process` — progressive ball-carrying & receiving &nbsp;|&nbsp;"
            "`② Defense Process` — high-press & interceptions &nbsp;|&nbsp;"
            "`③ Goal Threat` — involvement in shots &nbsp;|&nbsp;"
            "`④ Save Contrib` — saves, blocks, minus turnovers &nbsp;|&nbsp;"
            "`Critical Total` — ③ + ④"
        )


    render_sb_footer()


# =========================================================
# PAGE B — Tournament Rankings
# =========================================================
elif page == "🏆 Tournament Rankings":

    st.subheader("🏆 2022 FIFA World Cup — Full Tournament Player Rankings")
    st.info(
        "All 64 matches scored on a single WC-wide scale.  \n"
        "Players appearing in multiple matches accumulate scores across games.  \n"
        "Use the sidebar filter to narrow by team name."
    )

    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "🥇 Overall Process", "⚡ ① Attack", "🛡️ ② Defense", "🎯 ③④ Critical Moments"
    ])

    team_filter = st.sidebar.text_input("Filter by team (rankings)", "")

    def get_ranked(col_jp, top_n=50):
        col_en = EN_LABELS.get(col_jp, col_jp)
        df = pool_df[["player","team", col_jp]].copy()
        if team_filter:
            df = df[df["team"].str.lower().str.contains(team_filter.lower())]
        df_agg = df.groupby(["player","team"])[col_jp].sum().reset_index()
        df_agg = df_agg.sort_values(col_jp, ascending=False).head(top_n)
        df_agg["Rank"] = range(1, len(df_agg)+1)
        df_agg = df_agg.rename(columns={"player":"Player","team":"Team", col_jp: col_en})
        return df_agg[["Rank","Player","Team", col_en]]

    with rank_tab1:
        st.caption("Attack + Defense combined across all matches played.")
        df_r = get_ranked("総合プロセス(①+②)")
        col_en = EN_LABELS["総合プロセス(①+②)"]
        st.dataframe(
            df_r.style.background_gradient(subset=[col_en], cmap="RdYlGn")
            .format({col_en: "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab2:
        st.caption("Players who most consistently drove the ball into dangerous areas via dribbles and progressive pass receiving.")
        df_r2 = get_ranked("①攻撃プロセス")
        col_en2 = EN_LABELS["①攻撃プロセス"]
        st.dataframe(
            df_r2.style.background_gradient(subset=[col_en2], cmap="YlGn")
            .format({col_en2: "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab3:
        st.caption("Players who most consistently won the ball in advanced areas (high press) and via interceptions.")
        df_r3 = get_ranked("②守備プロセス")
        col_en3 = EN_LABELS["②守備プロセス"]
        st.dataframe(
            df_r3.style.background_gradient(subset=[col_en3], cmap="Blues")
            .format({col_en3: "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab4:
        st.caption("Players most directly involved in shot creation (③) and shot prevention (④) — combined.")
        df_r4 = get_ranked("総合クリティカル(③+④)")
        col_en4 = EN_LABELS["総合クリティカル(③+④)"]
        st.dataframe(
            df_r4.style.background_gradient(subset=[col_en4], cmap="Purples")
            .format({col_en4: "{:+.3f}"}),
            use_container_width=True
        )

    render_sb_footer()
