import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

st.set_page_config(page_title="onFlows Demo – Zone Dynamics", layout="wide")

st.title("onFlows Demo: Динамика по зони × тренировъчни средства + индекс на стрес (ACWR)")
st.caption("Демо приложение със синтетични данни (1 ред = 1 минута). Цел: да визуализира желаната отчетност за федерация/отбор.")

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name="raw_minute")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

def zone_from_hr(hr: float, lthr: int) -> int:
    z1 = 0.75 * lthr
    z2 = 0.85 * lthr
    z3 = 0.92 * lthr
    z4 = 1.00 * lthr
    if hr < z1: return 1
    if hr < z2: return 2
    if hr < z3: return 3
    if hr < z4: return 4
    return 5

def compute_load(df_minute: pd.DataFrame, zone_weights: dict[int, float]) -> pd.DataFrame:
    """Daily load = sum(zone_weight * minutes)."""
    d = df_minute.copy()
    d["minutes"] = d["duration_sec"] / 60.0
    d["load"] = d["zone"].map(zone_weights) * d["minutes"]
    daily = d.groupby(["date"], as_index=False).agg(
        total_minutes=("minutes", "sum"),
        total_load=("load", "sum"),
        avg_hr=("hr_bpm", "mean"),
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    # rolling acute/chronic
    daily["acute_7d"] = daily["total_load"].rolling(7, min_periods=1).sum()
    daily["chronic_28d"] = daily["total_load"].rolling(28, min_periods=1).sum()
    daily["stress_index_acwr"] = daily["acute_7d"] / daily["chronic_28d"].replace(0, np.nan)
    return daily

uploaded = st.file_uploader("Качи Excel (sheet: raw_minute). Можеш да ползваш демо файла, който ти генерирахме.", type=["xlsx"])

if not uploaded:
    st.info("Качи Excel файл, за да стартира демото.")
    st.stop()

df = load_data(uploaded)

# Sidebar controls
with st.sidebar:
    st.header("Настройки")
    athletes = sorted(df["athlete_id"].unique().tolist())
    athlete = st.selectbox("Атлет", athletes, index=0)
    df_a = df[df["athlete_id"] == athlete].copy()

    st.subheader("HR праг и зони")
    lthr_default = int(df_a["hr_bpm"].quantile(0.95))  # rough guess for demo
    lthr = st.slider("LTHR (праг HR) [bpm]", min_value=130, max_value=200, value=min(170, max(130, lthr_default)))
    st.caption("Зони: Z1<75% LTHR, Z2 75–85, Z3 85–92, Z4 92–100, Z5>100%")

    st.subheader("Тегла за натоварване по зона")
    w1 = st.number_input("Z1 тегло", value=1.0, step=0.1)
    w2 = st.number_input("Z2 тегло", value=2.0, step=0.1)
    w3 = st.number_input("Z3 тегло", value=3.0, step=0.1)
    w4 = st.number_input("Z4 тегло", value=4.0, step=0.1)
    w5 = st.number_input("Z5 тегло", value=5.0, step=0.1)
    zone_weights = {1:w1, 2:w2, 3:w3, 4:w4, 5:w5}

    st.subheader("Период")
    min_d, max_d = pd.to_datetime(df_a["date"].min()), pd.to_datetime(df_a["date"].max())
    date_range = st.date_input("От–До", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
    if isinstance(date_range, tuple):
        d0, d1 = date_range
    else:
        d0, d1 = min_d.date(), max_d.date()

df_a = df[df["athlete_id"] == athlete].copy()
df_a["timestamp"] = pd.to_datetime(df_a["timestamp"])
df_a = df_a[(df_a["timestamp"].dt.date >= d0) & (df_a["timestamp"].dt.date <= d1)].copy()

# Compute zones
df_a["zone"] = df_a["hr_bpm"].apply(lambda x: zone_from_hr(x, lthr))
df_a["minutes"] = df_a["duration_sec"] / 60.0

# ----------------------------
# 1) Zone × Sport aggregation
# ----------------------------
agg = df_a.groupby(["sport", "zone"], as_index=False).agg(
    minutes=("minutes", "sum"),
    avg_hr=("hr_bpm", "mean"),
    sessions=("session_id", pd.Series.nunique),
)
agg["hours"] = agg["minutes"] / 60.0

st.subheader("Разпределение по тренировъчни средства × зони (за избрания период)")
c1, c2 = st.columns([1.25, 1])
with c1:
    # Heatmap-like chart: sport (y) x zone (x) with hours
    heat = alt.Chart(agg).mark_rect().encode(
        x=alt.X("zone:O", title="Зона"),
        y=alt.Y("sport:N", title="Средство", sort="-x"),
        tooltip=[
            alt.Tooltip("sport:N", title="Средство"),
            alt.Tooltip("zone:O", title="Зона"),
            alt.Tooltip("hours:Q", title="Часове", format=".2f"),
            alt.Tooltip("avg_hr:Q", title="Среден HR (в зоната)", format=".0f"),
            alt.Tooltip("sessions:Q", title="Сесии"),
        ],
        color=alt.Color("hours:Q", title="Часове"),
    ).properties(height=320)
    st.altair_chart(heat, use_container_width=True)

with c2:
    st.dataframe(
        agg.sort_values(["sport","zone"])[["sport","zone","hours","avg_hr","sessions"]],
        use_container_width=True,
        hide_index=True
    )
    st.caption("Това е форматът, който TrainingPeaks по подразбиране НЕ дава като агрегирана таблица със среден HR в зоната за периода.")

# ----------------------------
# 2) Weekly (or daily) dynamics by zone
# ----------------------------
st.subheader("Динамика по седмици: време в зона + среден HR в зона")
df_a["date"] = pd.to_datetime(df_a["date"])
df_a["week"] = df_a["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

weekly = df_a.groupby(["week","sport","zone"], as_index=False).agg(
    minutes=("minutes","sum"),
    avg_hr=("hr_bpm","mean"),
)
weekly["hours"] = weekly["minutes"] / 60.0

sport_sel = st.multiselect("Филтър средство (по желание)", sorted(df_a["sport"].unique().tolist()), default=sorted(df_a["sport"].unique().tolist()))
weekly_f = weekly[weekly["sport"].isin(sport_sel)].copy()

# stacked area: hours by zone over weeks (sum across sports filter)
weekly_zone = weekly_f.groupby(["week","zone"], as_index=False).agg(hours=("hours","sum"))
area = alt.Chart(weekly_zone).mark_area().encode(
    x=alt.X("week:T", title="Седмица"),
    y=alt.Y("hours:Q", title="Часове (общо)"),
    color=alt.Color("zone:O", title="Зона"),
    tooltip=[alt.Tooltip("week:T", title="Седмица"), alt.Tooltip("zone:O", title="Зона"), alt.Tooltip("hours:Q", title="Часове", format=".2f")]
).properties(height=280)
st.altair_chart(area, use_container_width=True)

# line: avg HR by zone over weeks (weighted by minutes)
weekly_hr = weekly_f.copy()
weekly_hr["hr_x_min"] = weekly_hr["avg_hr"] * weekly_hr["minutes"]
tmp = weekly_hr.groupby(["week","zone"], as_index=False).agg(
    minutes=("minutes","sum"),
    hr_x_min=("hr_x_min","sum"),
)
tmp["avg_hr_in_zone"] = tmp["hr_x_min"] / tmp["minutes"].replace(0, np.nan)

line = alt.Chart(tmp).mark_line(point=True).encode(
    x=alt.X("week:T", title="Седмица"),
    y=alt.Y("avg_hr_in_zone:Q", title="Среден HR в зоната (седмично)"),
    color=alt.Color("zone:O", title="Зона"),
    tooltip=[
        alt.Tooltip("week:T", title="Седмица"),
        alt.Tooltip("zone:O", title="Зона"),
        alt.Tooltip("avg_hr_in_zone:Q", title="Среден HR", format=".0f"),
        alt.Tooltip("minutes:Q", title="Минути", format=".0f"),
    ]
).properties(height=280)

st.altair_chart(line, use_container_width=True)

# ----------------------------
# 3) Stress index (ACWR-like)
# ----------------------------
st.subheader("Индекс на стрес (ACWR-like): Acute 7 дни / Chronic 28 дни")
daily = df_a.groupby(["date","zone"], as_index=False).agg(minutes=("minutes","sum"))
daily["load"] = daily["zone"].map(zone_weights) * daily["minutes"]

daily_tot = daily.groupby("date", as_index=False).agg(
    minutes=("minutes","sum"),
    load=("load","sum"),
)
daily_tot = daily_tot.sort_values("date")
daily_tot["acute_7d"] = daily_tot["load"].rolling(7, min_periods=1).sum()
daily_tot["chronic_28d"] = daily_tot["load"].rolling(28, min_periods=1).sum()
daily_tot["acwr"] = daily_tot["acute_7d"] / daily_tot["chronic_28d"].replace(0, np.nan)

c3, c4 = st.columns([1.4, 1])
with c3:
    acwr_chart = alt.Chart(daily_tot).transform_fold(
        ["load","acute_7d","chronic_28d","acwr"],
        as_=["metric","value"]
    ).mark_line().encode(
        x=alt.X("date:T", title="Дата"),
        y=alt.Y("value:Q", title="Стойност"),
        color=alt.Color("metric:N", title="Метрика"),
        tooltip=[alt.Tooltip("date:T", title="Дата"), alt.Tooltip("metric:N", title="Метрика"), alt.Tooltip("value:Q", title="Стойност", format=".2f")]
    ).properties(height=320)
    st.altair_chart(acwr_chart, use_container_width=True)

with c4:
    st.dataframe(daily_tot.tail(14), use_container_width=True, hide_index=True)
    st.caption("ACWR тук е демонстрационен (зонално-теглови load). В реалния проект може да се смени с TRIMP, sRPE, HRV-модел, CS/W′ и т.н.")

st.divider()
st.subheader("Експорт на агрегирания отчет (CSV)")
# build the exact table format the user described: period × sport × zone → hours + avg HR
report = weekly_f.copy()
report_out = report[["week","sport","zone","hours","avg_hr"]].copy()
report_out = report_out.sort_values(["week","sport","zone"])
csv = report_out.to_csv(index=False).encode("utf-8")
st.download_button("Свали седмичен отчет (CSV)", csv, file_name="weekly_zone_report.csv", mime="text/csv")
