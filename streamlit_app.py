import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="onFlows Demo – Zone Dynamics", layout="wide")

st.title("onFlows Demo: Динамика по зони × тренировъчни средства + индекс на стрес (ACWR)")
st.caption("Демо приложение: агрегиране по средства и зони + динамика + ACWR-like. 1 ред = 1 минута.")

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name="raw_minute")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
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

uploaded = st.file_uploader("Качи Excel (sheet: raw_minute).", type=["xlsx"])

if not uploaded:
    st.info("Качи Excel файл, за да стартира демото.")
    st.stop()

df = load_data(uploaded)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Настройки")
    athletes = sorted(df["athlete_id"].unique().tolist())
    athlete = st.selectbox("Атлет", athletes, index=0)

    df_a0 = df[df["athlete_id"] == athlete].copy()

    st.subheader("HR праг и зони")
    lthr_guess = int(df_a0["hr_bpm"].quantile(0.95))
    lthr = st.slider("LTHR (праг HR) [bpm]", 130, 205, value=min(172, max(130, lthr_guess)))
    st.caption("Зони: Z1<75% LTHR, Z2 75–85, Z3 85–92, Z4 92–100, Z5>100%")

    st.subheader("Натоварване (Load) – метод")
    load_method = st.radio(
        "Избор на модел за дневно натоварване",
        options=[
            "HR-based: minutes × avgHR (ден)",
            "HR-based normalized: minutes × (avgHR/LTHR)",
            "Zone-weighted: minutes × zone_weight",
        ],
        index=0
    )

    st.subheader("Тегла за зони (ползват се само при Zone-weighted)")
    w1 = st.number_input("Z1 тегло", value=1.0, step=0.5)
    w2 = st.number_input("Z2 тегло", value=2.0, step=0.5)
    w3 = st.number_input("Z3 тегло", value=4.0, step=0.5)
    w4 = st.number_input("Z4 тегло", value=8.0, step=0.5)
    w5 = st.number_input("Z5 тегло", value=12.0, step=0.5)
    zone_weights = {1: w1, 2: w2, 3: w3, 4: w4, 5: w5}

    st.subheader("Период")
    min_d, max_d = df_a0["date"].min(), df_a0["date"].max()
    date_range = st.date_input(
        "От–До",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date()
    )
    if isinstance(date_range, tuple):
        d0, d1 = date_range
    else:
        d0, d1 = min_d.date(), max_d.date()

# Filter athlete + period
df_a = df[df["athlete_id"] == athlete].copy()
df_a = df_a[(df_a["date"].dt.date >= d0) & (df_a["date"].dt.date <= d1)].copy()

# Compute zones + minutes
df_a["zone"] = df_a["hr_bpm"].apply(lambda x: zone_from_hr(x, lthr))
df_a["minutes"] = df_a["duration_sec"] / 60.0

# ----------------------------
# 1) Sport × Zone summary (period)
# ----------------------------
agg = df_a.groupby(["sport", "zone"], as_index=False).agg(
    minutes=("minutes", "sum"),
    avg_hr=("hr_bpm", "mean"),
    sessions=("session_id", pd.Series.nunique),
)
agg["hours"] = agg["minutes"] / 60.0

st.subheader("Разпределение по тренировъчни средства × зони (за избрания период)")

heat = alt.Chart(agg).mark_rect().encode(
    x=alt.X("zone:O", title="Зона"),
    y=alt.Y("sport:N", title="Средство"),
    tooltip=[
        alt.Tooltip("sport:N", title="Средство"),
        alt.Tooltip("zone:O", title="Зона"),
        alt.Tooltip("hours:Q", title="Часове", format=".2f"),
        alt.Tooltip("avg_hr:Q", title="Среден HR (в зоната)", format=".0f"),
        alt.Tooltip("sessions:Q", title="Сесии"),
    ],
    color=alt.Color("hours:Q", title="Часове"),
).properties(height=320)

c1, c2 = st.columns([1.25, 1])
with c1:
    st.altair_chart(heat, use_container_width=True)
with c2:
    st.dataframe(
        agg.sort_values(["sport", "zone"])[["sport", "zone", "hours", "avg_hr", "sessions"]],
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------
# 2) Weekly dynamics: time in zone + avg HR in zone
# ----------------------------
st.subheader("Динамика по седмици: време в зона + среден HR в зона")

df_a["week"] = df_a["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

weekly = df_a.groupby(["week", "sport", "zone"], as_index=False).agg(
    minutes=("minutes", "sum"),
    avg_hr=("hr_bpm", "mean"),
)
weekly["hours"] = weekly["minutes"] / 60.0

sport_sel = st.multiselect(
    "Филтър средство",
    sorted(df_a["sport"].unique().tolist()),
    default=sorted(df_a["sport"].unique().tolist()),
)
weekly_f = weekly[weekly["sport"].isin(sport_sel)].copy()

weekly_zone = weekly_f.groupby(["week", "zone"], as_index=False).agg(hours=("hours", "sum"))
area = alt.Chart(weekly_zone).mark_area().encode(
    x=alt.X("week:T", title="Седмица"),
    y=alt.Y("hours:Q", title="Часове (общо)"),
    color=alt.Color("zone:O", title="Зона"),
    tooltip=[
        alt.Tooltip("week:T", title="Седмица"),
        alt.Tooltip("zone:O", title="Зона"),
        alt.Tooltip("hours:Q", title="Часове", format=".2f"),
    ],
).properties(height=280)
st.altair_chart(area, use_container_width=True)

# weekly avg HR in zone (weighted by minutes)
weekly_hr = weekly_f.copy()
weekly_hr["hr_x_min"] = weekly_hr["avg_hr"] * weekly_hr["minutes"]
tmp = weekly_hr.groupby(["week", "zone"], as_index=False).agg(
    minutes=("minutes", "sum"),
    hr_x_min=("hr_x_min", "sum"),
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
    ],
).properties(height=280)
st.altair_chart(line, use_container_width=True)

# ----------------------------
# 3) Stress index (ACWR-like) – CORRECT rolling logic
#   acute_7d = mean load last 7 days
#   chronic_28d = mean load last 28 days
# ----------------------------
st.subheader("Индекс на стрес (ACWR-like): Acute 7 дни / Chronic 28 дни")

# daily aggregates needed for HR-based load
daily_hr = df_a.groupby("date", as_index=False).agg(
    minutes=("minutes", "sum"),
    avg_hr=("hr_bpm", "mean"),
)

# define daily load based on selected method
if load_method == "HR-based: minutes × avgHR (ден)":
    daily_hr["load"] = daily_hr["minutes"] * daily_hr["avg_hr"]
elif load_method == "HR-based normalized: minutes × (avgHR/LTHR)":
    daily_hr["load"] = daily_hr["minutes"] * (daily_hr["avg_hr"] / float(lthr))
else:
    # zone-weighted load
    daily_zone = df_a.groupby(["date", "zone"], as_index=False).agg(minutes=("minutes", "sum"))
    daily_zone["load_piece"] = daily_zone["zone"].map(zone_weights) * daily_zone["minutes"]
    daily_hr = daily_zone.groupby("date", as_index=False).agg(
        minutes=("minutes", "sum"),
        load=("load_piece", "sum"),
    )
    # keep avg_hr column optional
    daily_hr["avg_hr"] = np.nan

daily_hr = daily_hr.sort_values("date")

# Rolling means (this is the key fix)
daily_hr["acute_7d"] = daily_hr["load"].rolling(7, min_periods=1).mean()
daily_hr["chronic_28d"] = daily_hr["load"].rolling(28, min_periods=1).mean()
daily_hr["acwr"] = daily_hr["acute_7d"] / daily_hr["chronic_28d"].replace(0, np.nan)

# Chart
chart = alt.Chart(daily_hr).transform_fold(
    ["load", "acute_7d", "chronic_28d", "acwr"],
    as_=["metric", "value"]
).mark_line().encode(
    x=alt.X("date:T", title="Дата"),
    y=alt.Y("value:Q", title="Стойност"),
    color=alt.Color("metric:N", title="Метрика"),
    tooltip=[
        alt.Tooltip("date:T", title="Дата"),
        alt.Tooltip("metric:N", title="Метрика"),
        alt.Tooltip("value:Q", title="Стойност", format=".2f"),
    ]
).properties(height=360)

st.altair_chart(chart, use_container_width=True)
st.dataframe(daily_hr.tail(21), use_container_width=True, hide_index=True)

# Export weekly report CSV (as before)
st.divider()
st.subheader("Експорт: седмичен отчет (CSV)")

report_out = weekly_f[["week", "sport", "zone", "hours", "avg_hr"]].copy()
report_out = report_out.sort_values(["week", "sport", "zone"])
csv = report_out.to_csv(index=False).encode("utf-8")
st.download_button("Свали седмичен отчет (CSV)", csv, file_name="weekly_zone_report.csv", mime="text/csv")
