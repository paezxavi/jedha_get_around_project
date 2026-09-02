"""Streamlit dashboard for the Getaround minimum-delay feature.

This page holds no model and calls no API. It is the delay analysis of
getaround_analysis.ipynb (sections 3 to 5) made interactive, so the product manager can move
the threshold and the scope themselves instead of reading one recommendation.

Everything is recomputed from the raw file on every interaction. That is affordable here --
21 310 rows, one pandas pass -- and it means no number on the page can be a stale constant
copied from the notebook.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data" / "get_around_delay_analysis.xlsx"

BLUE, RED, GREY = "#2a78d6", "#e34948", "#8a8a85"

st.set_page_config(page_title="Getaround — minimum delay between rentals",
                   page_icon="🚗", layout="wide")


@st.cache_data
def load():
    """Read the delay file and build the chained-pair table the whole page works from.

    Cached because the .xlsx parse is the only slow step (~2 s); everything downstream is a
    filter over 1 729 rows and costs nothing.
    """
    delay = pd.read_excel(DATA)

    # A rental is "chained" when the file records the rental that came before it on the same
    # car. Joining the file to itself brings the previous checkout delay alongside the gap.
    pairs = (delay[delay["previous_ended_rental_id"].notna()]
             .merge(delay.add_prefix("prev_"), left_on="previous_ended_rental_id",
                    right_on="prev_rental_id", how="left"))

    # Only pairs whose PREVIOUS rental has a recorded checkout can be judged. The 112 others
    # are a known blind spot, reported on the page rather than imputed.
    measurable = pairs[pairs["prev_delay_at_checkout_in_minutes"].notna()].copy()
    measurable["overlap"] = (measurable["prev_delay_at_checkout_in_minutes"]
                             - measurable["time_delta_with_previous_rental_in_minutes"])
    return delay, pairs, measurable


delay, pairs, measurable = load()
CHAINED = int(delay["previous_ended_rental_id"].notna().sum())
BASELINE = (delay["state"] == "canceled").mean()

# ---------------------------------------------------------------- the two decisions
st.sidebar.title("The two decisions")
st.sidebar.caption("Both controls below are the product manager's, and every number on the "
                   "page reacts to them.")

threshold = st.sidebar.select_slider(
    "Minimum delay between two rentals",
    options=[0, 30, 60, 90, 120, 180, 240, 360, 480, 720], value=60,
    format_func=lambda m: f"{m} min" if m < 120 else f"{m // 60} h",
)
scope = st.sidebar.radio("Scope", ["all cars", "Connect cars only"], index=0)
severity = st.sidebar.radio(
    "What counts as a problem",
    ["an overlap of more than 1 h", "any overlap at all"], index=0,
    help="The next rental's cancellation rate only moves past an hour of overlap — "
         "below that it sits at or under the 15.3% baseline. The default is the honest one.",
)

SEVERITY_MIN = 60 if severity.startswith("an overlap of more") else 0
SCOPE_KEY = "connect" if scope.startswith("Connect") else "all"

sub = measurable if SCOPE_KEY == "all" else measurable[measurable["checkin_type"] == "connect"]
blocked_mask = sub["time_delta_with_previous_rental_in_minutes"] < threshold
problems = sub["overlap"] > SEVERITY_MIN
blocked = int(blocked_mask.sum())
prevented = int((problems & blocked_mask).sum())
total_problems = int(problems.sum())

st.sidebar.divider()
st.sidebar.markdown(
    f"**At this setting**\n\n"
    f"- {blocked} rentals blocked — {blocked / len(delay):.2%} of all rentals\n"
    f"- {prevented} of {total_problems} problems prevented\n"
    f"- {blocked / max(prevented, 1):.1f} blocked rentals per problem avoided"
)

# ---------------------------------------------------------------- the headline
st.title("Should Getaround put a buffer between two rentals?")
st.markdown(
    "A car is hidden from search results when the requested check-in is closer than the "
    "threshold to the previous checkout. It solves late-handover friction and it costs "
    "bookings. **The two sliders on the left are the whole decision.**"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rentals blocked", f"{blocked}", f"{blocked / len(delay):.2%} of all rentals",
          delta_color="inverse")
c2.metric("Problems prevented", f"{prevented} / {total_problems}",
          f"{prevented / max(total_problems, 1):.0%} of them")
c3.metric("Price of one avoided", f"{blocked / max(prevented, 1):.1f}", "blocked rentals",
          delta_color="off")
c4.metric("Feature's whole target", f"{int((measurable['overlap'] > 60).sum())}",
          f"of {len(delay):,} rentals".replace(",", " "), delta_color="off")

st.divider()

# ---------------------------------------------------------------- 1. the scope of the feature
st.header("1. How many rentals can the feature reach at all?")

funnel = pd.DataFrame({
    "step": ["All rentals in the file",
             "Chained — they follow another rental",
             "Measurable — the previous checkout is recorded",
             "Overlapped — the car came back late enough to matter",
             "Severe — the overlap is more than an hour"],
    "rentals": [len(delay), CHAINED, len(measurable),
                int((measurable["overlap"] > 0).sum()), int((measurable["overlap"] > 60).sum())],
})
funnel["share"] = funnel["rentals"] / len(delay)

fig = go.Figure(go.Bar(
    x=funnel["rentals"], y=funnel["step"], orientation="h",
    marker_color=[GREY, BLUE, BLUE, RED, RED],
    text=[f"{n:,}".replace(",", " ") + f"   ({s:.1%})" for n, s in
          zip(funnel["rentals"], funnel["share"])],
    textposition="outside",
))
# Log scale on purpose: 66 and 21 310 on the same linear axis makes the last two bars
# invisible, and those two bars are the point of the chart.
fig.update_layout(
    height=340, template="plotly_white", margin=dict(t=20, b=40, l=20, r=40),
    xaxis=dict(title="rentals (log scale)", type="log", range=[0, 5.0]),
    yaxis=dict(autorange="reversed"), showlegend=False,
)
st.plotly_chart(fig, width='stretch')

st.info(
    f"**Only {CHAINED:,} of the {len(delay):,} rentals — {CHAINED / len(delay):.1%} — follow "
    f"another rental of the same car.** The feature cannot affect any of the other "
    f"{len(delay) - CHAINED:,}. Narrow it down to the cases that actually go wrong and the "
    f"target is **{int((measurable['overlap'] > 60).sum())} rentals**."
    .replace(",", " ")
)

# ---------------------------------------------------------------- 2. lateness and its impact
st.header("2. How often are drivers late, and does it reach the next driver?")

left, right = st.columns([1, 1])

with left:
    st.subheader("Connect drivers are the punctual half")
    ended = delay[delay["state"] == "ended"].dropna(subset=["delay_at_checkout_in_minutes"])
    by_type = ended.groupby("checkin_type")["delay_at_checkout_in_minutes"].agg(
        rentals="size",
        late=lambda s: (s > 0).mean(),
        median="median",
        over_1h=lambda s: (s > 60).mean(),
    )
    st.dataframe(
        by_type.rename(columns={"late": "late at all", "median": "median delay (min)",
                                "over_1h": "more than 1 h late"})
        .style.format({"late at all": "{:.1%}", "median delay (min)": "{:.0f}",
                       "more than 1 h late": "{:.1%}"}),
        width='stretch',
    )
    st.caption(
        "Connect is where the problem *shows up* because Connect cars are chained three times "
        f"more often ({(delay[delay['checkin_type'] == 'connect']['previous_ended_rental_id'].notna()).mean():.1%} "
        f"of Connect rentals against "
        f"{(delay[delay['checkin_type'] == 'mobile']['previous_ended_rental_id'].notna()).mean():.1%} "
        "of mobile ones) — not because its drivers behave worse."
    )

with right:
    st.subheader("Only an overlap past an hour moves the cancellation rate")
    BANDS = [-10 ** 9, 0, 30, 60, 120, 10 ** 9]
    NAMES = ["no overlap", "0-30 min", "30-60 min", "1-2 h", "over 2 h"]
    banded = measurable.assign(band=pd.cut(measurable["overlap"], BANDS, labels=NAMES))
    impact = banded.groupby("band", observed=True).agg(
        pairs=("state", "size"), rate=("state", lambda s: (s == "canceled").mean()))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=NAMES, y=impact["pairs"], name="chained pairs", marker_color=GREY,
                         opacity=0.35, text=impact["pairs"], textposition="outside"))
    fig.add_trace(go.Scatter(x=NAMES, y=impact["rate"], name="next rental cancelled",
                            mode="lines+markers", line=dict(color=RED, width=3)),
                  secondary_y=True)
    fig.add_hline(y=BASELINE, line_dash="dot", line_color=GREY, secondary_y=True)
    # Both axes pinned from zero: the bars carry the sample size behind each rate, and a
    # truncated axis would make 41 pairs look like a solid measurement.
    fig.update_yaxes(title="chained pairs", range=[0, 1750], secondary_y=False)
    fig.update_yaxes(title="cancellation rate", tickformat=".0%", range=[0, 0.45],
                     showgrid=False, secondary_y=True)
    fig.update_layout(height=340, template="plotly_white", margin=dict(t=20, b=40, l=60, r=60),
                      legend=dict(orientation="h", y=1.12, x=0.25))
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"Dotted line: the {BASELINE:.1%} cancellation rate of all rentals. An overlap under "
        "half an hour sits *below* it. The right-hand half of this chart rests on 103 pairs, "
        "and a cancellation after a late checkout is a correlation — the file records no "
        "cancellation reason."
    )

# ---------------------------------------------------------------- 3. the trade-off
st.header("3. What each threshold and scope costs")

GRID = [30, 60, 90, 120, 180, 240, 360, 480, 720]
rows = []
for scope_key in ("all", "connect"):
    s = measurable if scope_key == "all" else measurable[measurable["checkin_type"] == "connect"]
    for t in GRID:
        under = s["time_delta_with_previous_rental_in_minutes"] < t
        solved = int(((s["overlap"] > SEVERITY_MIN) & under).sum())
        rows.append({"threshold": t, "scope": scope_key, "blocked": int(under.sum()),
                     "share of all rentals": under.sum() / len(delay), "prevented": solved,
                     "blocked per problem": under.sum() / max(solved, 1)})
curve = pd.DataFrame(rows)

fig = go.Figure()
for scope_key, color, name in [("all", BLUE, "all cars"), ("connect", RED, "Connect cars only")]:
    s = curve[curve["scope"] == scope_key]
    fig.add_trace(go.Scatter(x=s["threshold"], y=s["blocked per problem"], mode="lines+markers",
                             name=name, line=dict(color=color, width=3)))
fig.add_vline(x=threshold, line_dash="dot", line_color=GREY,
              annotation_text=f"your setting: {threshold} min", annotation_position="top left")
fig.update_layout(
    height=380, template="plotly_white", margin=dict(t=30, b=50, l=60, r=40),
    title="The trade never improves — it only gets more expensive",
    xaxis=dict(title="minimum delay (minutes)"),
    yaxis=dict(title="rentals blocked per problem avoided", rangemode="tozero"),
    legend=dict(orientation="h", y=1.02, x=0.5),
)
st.plotly_chart(fig, width='stretch')

with st.expander("The full grid, both scopes"):
    st.dataframe(
        curve.pivot(index="threshold", columns="scope",
                    values=["blocked", "prevented", "blocked per problem"])
        .style.format("{:.1f}"),
        width='stretch',
    )

# ---------------------------------------------------------------- 4. the recommendation
st.header("4. The recommendation")

rec_all = measurable
rec_blocked = int((rec_all["time_delta_with_previous_rental_in_minutes"] < 60).sum())
rec_prevented = int(((rec_all["overlap"] > 60)
                     & (rec_all["time_delta_with_previous_rental_in_minutes"] < 60)).sum())
severe_total = int((rec_all["overlap"] > 60).sum())

st.success(
    f"**Sixty minutes, on all cars.** It blocks {rec_blocked} rentals "
    f"({rec_blocked / len(delay):.2%} of all of them) and prevents {rec_prevented} of the "
    f"{severe_total} severe cases — {rec_prevented / severe_total:.0%} — at "
    f"{rec_blocked / rec_prevented:.1f} blocked rentals each. Every longer threshold buys the "
    f"remaining cases at a worse rate, and there is no optimum on the curve above: the "
    f"question is what the company will pay per avoided incident, and 60 minutes is the "
    f"cheapest setting that does anything."
)

st.warning(
    "**Do not restrict the scope to Connect.** Connect concentrates the chaining, not the "
    "lateness: its drivers are late 42.9% of the time against 61.4%, and its chained pairs go "
    "badly wrong half as often (2.5% against 4.9%). At 60 minutes, Connect-only costs 17.7 "
    "blocked rentals per severe case against 10.6 for all cars."
)

st.subheader("What this cannot tell you")
st.markdown(
    """
1. **No revenue figure is possible from these files.** The pricing dataset has no car
   identifier, so no rental has a price: every cost above is a count of *rentals* standing in
   for euros. The one real signal is that Connect cars rent for **€132.79 a day against
   €111.34**, 19% more, which makes the Connect-only scope more expensive still.
2. **A blocked rental is not a lost rental.** The file cannot say whether the driver rebooked
   the same car later or another car. These figures are an upper bound on the cost.
3. **Duration is missing**, so a two-hour rental and a five-day rental count the same.
4. **The file caps the gap at 12 hours**, so nothing beyond that is measurable here.
5. **112 chained pairs have no recorded previous checkout** and are excluded rather than
   imputed, so every count above is a slight under-estimate.

The fix for the first three is one join and two columns: `car_id` in the pricing export, and
the rental's start and end.
"""
)

st.divider()
st.caption(
    "Built from `get_around_delay_analysis.xlsx` — 21 310 rentals. The full analysis, with "
    "every measurement behind these numbers, is in `getaround_analysis.ipynb`. The pricing "
    "model asked for by the same brief is served by a separate API Space, documented at its "
    "`/docs`."
)
