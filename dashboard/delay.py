"""Delay analysis page: the minimum delay between two rentals.

The page answers the product manager's four questions, one section each, then puts cost (Q2)
against benefit (Q4) to recommend a threshold and a scope. It holds no model and calls no API;
getaround_analysis.ipynb is kept beside it as an annex.

Everything is recomputed from the raw file on every interaction. That is affordable here --
21 310 rows, one pandas pass -- and it means no number on the page can be a stale constant.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data" / "get_around_delay_analysis.xlsx"

BLUE, GREY = "#2a78d6", "#8a8a85"

# One colour per scope, kept the same in every chart so a scope is recognisable at a glance.
# None means "no filter on checkin_type".
SCOPES = {"all cars": None, "Connect cars only": "connect", "Mobile cars only": "mobile"}
SCOPE_COLORS = {"all cars": BLUE, "Connect cars only": "#eb6834", "Mobile cars only": "#1baf7a"}

# time_delta only takes multiples of 30 minutes, so the curves step every 30 minutes up to the
# 12-hour cap of the file. The recommended threshold is the one the last section argues for.
GAPS = list(range(0, 721, 30))
RECOMMENDED = 30


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

# Cost side (Q1, Q2): ended rentals, the only ones that earned anything.
is_ended = delay["state"] == "ended"
gap = delay["time_delta_with_previous_rental_in_minutes"]
is_exposed = is_ended & gap.notna()
# Benefit side (Q3, Q4): every pair with an overlap, whatever happened to the next rental.
pair_gap = measurable["time_delta_with_previous_rental_in_minutes"]
is_problem = measurable["overlap"] > 0


def in_scope(name):
    """Rows of a scope. A pair belongs to the checkin_type written on its own row, nothing else."""
    kind = SCOPES[name]
    return pd.Series(True, index=delay.index) if kind is None else delay["checkin_type"] == kind


def of_scope(table, name):
    """Rows of any table that carries checkin_type, kept to one scope."""
    kind = SCOPES[name]
    return table if kind is None else table[table["checkin_type"] == kind]


def cost_and_benefit(name, t):
    """Ended rentals blocked (Q2) and problematic cases solved (Q4) by threshold t in a scope."""
    blocked = int((is_ended & in_scope(name) & (gap < t)).sum())
    solved = int(of_scope(measurable[is_problem & (pair_gap < t)], name).shape[0])
    return blocked, solved


# ---------------------------------------------------------------- the two decisions
st.sidebar.title("The two decisions")
st.sidebar.caption("Both controls below are the product manager's, and every number on the "
                   "page reacts to them.")

threshold = st.sidebar.select_slider(
    "Minimum delay between two rentals",
    options=[0, 30, 60, 90, 120, 180, 240, 360, 480, 720], value=RECOMMENDED,
    format_func=lambda m: f"{m} min" if m < 120 else f"{m // 60} h",
)
scope = st.sidebar.radio("Scope", list(SCOPES), index=0)
# Charts show every scope when "all cars" is chosen, and only the chosen one otherwise.
SHOWN = list(SCOPES) if SCOPES[scope] is None else [scope]

side_blocked, side_solved = cost_and_benefit(scope, threshold)
side_problems = len(of_scope(measurable[is_problem], scope))
st.sidebar.divider()
st.sidebar.markdown(
    f"**At this setting — {scope}**\n\n"
    f"- {side_blocked} ended rentals blocked — {side_blocked / is_ended.sum():.2%} (Q2)\n"
    f"- {side_solved} of {side_problems} problematic cases solved (Q4)\n"
    f"- {side_blocked / max(side_solved, 1):.2f} ended rentals blocked per case solved"
)

# ---------------------------------------------------------------- the headline
st.title("Should Getaround put a buffer between two rentals?")
st.markdown(
    "A car is hidden from search results when a requested rental would start too soon after "
    "the previous one ends. It spares the next driver a car that is not back yet, and it costs "
    "bookings. The four sections below answer the product manager's four questions; the last "
    "one weighs cost against benefit. **The threshold and the scope on the left drive every "
    "number on the page.**"
)

st.divider()

# ---------------------------------------------------------------- Q1. revenue exposed
st.header("Q1. Which share of our owner's revenue would potentially be affected by the feature?")

# The rule only ever hides a car whose requested slot sits within the threshold of a previous
# rental, so a rental can be blocked only if it HAS a recorded gap. The file records that gap up
# to 12 hours: this is the ceiling of every threshold up to 12 h, which is why Q1 has none.
# Cancelled rentals are left out of both sides of the share: they never earned anything.
exposure = pd.DataFrame({"scope": SHOWN,
                         "rentals": [int((is_exposed & in_scope(s)).sum()) for s in SHOWN]})
exposure["share"] = exposure["rentals"] / is_ended.sum()
chosen = exposure.set_index("scope").loc[scope]

left, right = st.columns([1, 2])
left.metric(f"Revenue exposed — {scope}", f"{chosen['share']:.2%}",
            f"{int(chosen['rentals']):,} of {is_ended.sum():,} ended rentals".replace(",", " "),
            delta_color="off")

fig = go.Figure(go.Bar(
    x=exposure["share"], y=exposure["scope"], orientation="h",
    marker_color=[SCOPE_COLORS[s] for s in exposure["scope"]],
    text=[f"{s:.2%}   ({n:,} rentals)".replace(",", " ")
          for s, n in zip(exposure["share"], exposure["rentals"])],
    textposition="outside",
))
fig.update_layout(
    height=80 + 50 * len(SHOWN), template="plotly_white", margin=dict(t=10, b=40, l=20, r=40),
    xaxis=dict(title="share of ended rentals", tickformat=".0%",
               range=[0, exposure["share"].max() * 1.4]),
    yaxis=dict(autorange="reversed"), showlegend=False,
)
right.plotly_chart(fig, width='stretch')

st.caption(
    "**How this is counted.** A rental is *exposed* when it follows another rental of the same "
    "car by less than 12 hours (`time_delta_with_previous_rental_in_minutes` is filled): only "
    "those can ever be hidden, whatever the threshold. **Cancelled rentals are excluded** from "
    f"both sides of the share ({int((~is_ended).sum()):,} of them), because a rental that did "
    "not happen earned the owner nothing. **Every ended rental counts as the same revenue**: "
    "the file records neither a price nor a duration."
    .replace(",", " ")
)

st.divider()

# ---------------------------------------------------------------- Q2. rentals blocked
st.header("Q2. How many rentals would be affected by the feature depending on the threshold "
          "and scope we choose?")

# A rental is blocked when its planned gap to the previous rental is under the threshold; of the
# two rentals in a pair, the later one is the one the rule hides. Same rules as Q1: ended rentals
# only, and the scope read on the row itself. The curve does not move with the slider: it shows
# the trend.
curves = pd.DataFrame({s: [int((is_ended & in_scope(s) & (gap < t)).sum()) for t in GAPS]
                       for s in SHOWN}, index=GAPS)

# One tile per shown scope: with "all cars", Connect + Mobile add up to the first tile.
for col, s in zip(st.columns(len(SHOWN)), SHOWN):
    q2_blocked = int((is_ended & in_scope(s) & (gap < threshold)).sum())
    q2_exposed = int((is_exposed & in_scope(s)).sum())
    col.metric(f"Rentals blocked at {threshold} min — {s}",
               f"{q2_blocked:,} / {is_ended.sum():,}".replace(",", " "),
               f"{q2_blocked / is_ended.sum():.2%} of ended · "
               f"{q2_blocked / max(q2_exposed, 1):.1%} of exposed", delta_color="off")

fig = go.Figure()
for s in SHOWN:
    fig.add_trace(go.Scatter(x=curves.index, y=curves[s], mode="lines", name=s,
                             line=dict(color=SCOPE_COLORS[s], width=2)))
fig.add_vline(x=threshold, line_dash="dot", line_color=GREY,
              annotation_text=f"your setting: {threshold} min", annotation_position="top right")
fig.update_layout(
    height=380, template="plotly_white", margin=dict(t=30, b=50, l=60, r=40),
    xaxis=dict(title="minimum delay between two rentals (minutes)", dtick=60),
    yaxis=dict(title="ended rentals blocked", rangemode="tozero"),
    showlegend=len(SHOWN) > 1, legend=dict(orientation="h", y=1.08, x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, width='stretch')

st.caption(
    "**How this is counted.** A rental is *blocked* when the planned gap between its start and "
    "the previous rental's end is shorter than the threshold. In such a pair the rule hides the "
    "later rental, so the pair is counted once, on the later rental's row: its state (ended "
    "only, as in Q1) and its check-in type, as written in the file."
)

st.divider()

# ---------------------------------------------------------------- Q3. lateness and its impact
st.header("Q3. How often are drivers late for the next check-in? How does it impact the next "
          "driver?")

# Q3 describes the problem as it is today, so it ignores the threshold. "Late for the next
# check-in" means the previous driver's delay exceeds the planned gap: the overlap. It needs the
# previous checkout, so it is measured on the pairs where that checkout is recorded. Cancelled
# next rentals stay in: their cancellation IS the impact being measured.
returned = delay[is_ended & delay["delay_at_checkout_in_minutes"].notna()]

for col, s in zip(st.columns(len(SHOWN)), SHOWN):
    pairs_s, returned_s = of_scope(measurable, s), of_scope(returned, s)
    overlapping = int((pairs_s["overlap"] > 0).sum())
    col.metric(f"Late for the next check-in — {s}",
               f"{overlapping:,} / {len(pairs_s):,} pairs".replace(",", " "),
               f"{overlapping / len(pairs_s):.1%} of pairs · "
               f"{(returned_s['delay_at_checkout_in_minutes'] > 0).mean():.1%} of drivers late",
               delta_color="off")

# The impact chart is drawn for the chosen scope only: split three ways, the long-overlap bands
# fall to a dozen pairs each and their rates mean nothing.
q3_pairs = of_scope(measurable, scope)
BANDS = [-10 ** 9, 0, 30, 60, 120, 10 ** 9]
NAMES = ["no overlap", "0-30 min", "30-60 min", "1-2 h", "over 2 h"]
impact = (q3_pairs.assign(band=pd.cut(q3_pairs["overlap"], BANDS, labels=NAMES))
          .groupby("band", observed=False)
          .agg(pairs=("state", "size"), rate=("state", lambda x: (x == "canceled").mean())))
no_overlap_rate = impact.loc["no overlap", "rate"]
wait = q3_pairs.loc[q3_pairs["overlap"] > 0, "overlap"].median()

st.markdown(
    f"**Impact on the next driver — {scope}.** When the car is not back in time, the next "
    f"driver waits **{wait:.0f} minutes** (median). Below, how often the next rental ends up "
    f"cancelled, by how long the car came back after the next check-in was due."
)

fig = go.Figure(go.Bar(
    x=NAMES, y=impact["rate"],
    marker_color=[GREY] + [SCOPE_COLORS[scope]] * (len(NAMES) - 1),
    text=[f"{r:.1%} · {n} pairs" for r, n in zip(impact["rate"], impact["pairs"])],
    textposition="outside",
))
fig.add_hline(y=no_overlap_rate, line_dash="dot", line_color=GREY,
              annotation_text=f"no overlap: {no_overlap_rate:.1%}",
              annotation_position="top left")
fig.update_layout(
    height=400, template="plotly_white", margin=dict(t=50, b=50, l=60, r=40),
    title="Cancellation rate of the next rental, by overlap band",
    xaxis=dict(title="overlap: how late the car came back after the next check-in"),
    yaxis=dict(title="% of next rentals cancelled", tickformat=".0%",
               range=[0, impact["rate"].max() * 1.25]),
    showlegend=False,
)
st.plotly_chart(fig, width='stretch')

st.caption(
    f"**How this is counted.** Measured on the {len(measurable):,} chained pairs whose previous "
    f"checkout is recorded; {len(pairs) - len(measurable)} pairs have none and are left out "
    "rather than imputed. Drivers late = ended rentals returned after the planned end, among "
    "those with a recorded checkout. **Cancelled next rentals are kept here**, unlike Q1 and "
    "Q2, because the cancellation is the impact being measured. **This is a correlation**: the "
    "file records neither the reason nor the date of a cancellation, and the bands past half an "
    "hour rest on a few dozen pairs each."
    .replace(",", " ")
)

st.divider()

# ---------------------------------------------------------------- Q4. problems solved
st.header("Q4. How many problematic cases will it solve depending on the chosen threshold and "
          "scope?")

# A problematic case is a pair with an overlap, as in Q3, whatever happened to the next rental.
# It is solved when the rule would have hidden that next rental: its planned gap is under the
# threshold. Same curve and tiles as Q2, so cost and benefit read side by side.
solved_curves = pd.DataFrame(
    {s: [int(of_scope(measurable[is_problem & (pair_gap < t)], s).shape[0]) for t in GAPS]
     for s in SHOWN}, index=GAPS)

for col, s in zip(st.columns(len(SHOWN)), SHOWN):
    problems_s = of_scope(measurable[is_problem], s)
    solved_s = int((problems_s["time_delta_with_previous_rental_in_minutes"] < threshold).sum())
    col.metric(f"Problems solved at {threshold} min — {s}",
               f"{solved_s} / {len(problems_s)}",
               f"{solved_s / max(len(problems_s), 1):.1%} of problematic cases",
               delta_color="off")

fig = go.Figure()
for s in SHOWN:
    fig.add_trace(go.Scatter(x=solved_curves.index, y=solved_curves[s], mode="lines", name=s,
                             line=dict(color=SCOPE_COLORS[s], width=2)))
fig.add_vline(x=threshold, line_dash="dot", line_color=GREY,
              annotation_text=f"your setting: {threshold} min", annotation_position="top right")
fig.update_layout(
    height=380, template="plotly_white", margin=dict(t=30, b=50, l=60, r=40),
    xaxis=dict(title="minimum delay between two rentals (minutes)", dtick=60),
    yaxis=dict(title="problematic cases solved", rangemode="tozero"),
    showlegend=len(SHOWN) > 1, legend=dict(orientation="h", y=1.08, x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, width='stretch')

# Solved cases whose late return was longer than the buffer itself: if the blocked driver
# rebooked the same car at the first allowed slot, these would still overlap.
not_absorbed = int((of_scope(measurable[is_problem], scope)
                    .query("time_delta_with_previous_rental_in_minutes < @threshold"
                           " and prev_delay_at_checkout_in_minutes > @threshold")).shape[0])

st.caption(
    "**How this is counted.** A *problematic case* is a pair where the previous driver came "
    "back after the next check-in was due (an overlap, as in Q3). It is *solved* when the rule "
    "would have hidden the next rental: its planned gap is shorter than the threshold. **Unlike "
    "Q1 and Q2, cancelled next rentals are counted here**: a cancelled rental cost no revenue, "
    "but it can still be a problem the rule would have avoided. **Limit**: the file does not "
    "say what a blocked driver does next. If they rebooked the same car at the first allowed "
    f"slot, {not_absorbed} of the cases solved at {threshold} min ({scope}) would still overlap, "
    "because the previous driver was later than the buffer itself."
)

st.divider()

# ---------------------------------------------------------------- the recommendation
st.header("Recommendation")

# Cost (Q2) against benefit (Q4), one point per threshold. If a threshold stood out, the curve
# would bend sharply there; the marginal column of the table is the check that it does not.
trade = pd.DataFrame([(s, t, *cost_and_benefit(s, t)) for s in SHOWN for t in GAPS[1:]],
                     columns=["scope", "threshold", "blocked", "solved"])

fig = go.Figure()
for s in SHOWN:
    t_s = trade[trade["scope"] == s]
    fig.add_trace(go.Scatter(
        x=t_s["blocked"], y=t_s["solved"], mode="lines+markers", name=s,
        line=dict(color=SCOPE_COLORS[s], width=2), marker=dict(size=8),
        customdata=t_s["threshold"],
        hovertemplate="%{customdata} min: %{x} ended rentals blocked, "
                      "%{y} cases solved<extra>" + s + "</extra>",
    ))
    mine = t_s[t_s["threshold"] == threshold]
    fig.add_trace(go.Scatter(
        x=mine["blocked"], y=mine["solved"], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=GREY, width=2)),
    ))
fig.update_layout(
    height=420, template="plotly_white", margin=dict(t=50, b=50, l=60, r=40),
    title="Each point is a threshold: what it costs (Q2) against what it solves (Q4)",
    xaxis=dict(title="ended rentals blocked", rangemode="tozero"),
    yaxis=dict(title="problematic cases solved", rangemode="tozero"),
    showlegend=len(SHOWN) > 1, legend=dict(orientation="h", y=1.02, x=0),
)
st.plotly_chart(fig, width='stretch')
st.caption(f"Circled: your setting ({threshold} min). Hover a point for its threshold.")

with st.expander(f"The same trade, threshold by threshold — {scope}"):
    table = trade[trade["scope"] == scope].set_index("threshold")[["blocked", "solved"]]
    extra = table.diff().fillna(table)
    table["blocked per case solved"] = table["blocked"] / table["solved"]
    # The price of the LAST step: rentals blocked by moving up to this threshold, divided by
    # the cases that move solves. Noisy from step to step, never below the first step.
    # Written out as text: st.dataframe ignores the Styler's na_rep and would show "None"
    # on the steps that solve no extra case.
    table["extra blocked per extra case"] = [
        f"{b / c:.1f}" if c > 0 else "no extra case"
        for b, c in zip(extra["blocked"], extra["solved"])]
    st.dataframe(table.style.format({"blocked per case solved": "{:.2f}"}), width='stretch')

rec = {s: cost_and_benefit(s, RECOMMENDED) for s in SCOPES}
rec_problems = {s: len(of_scope(measurable[is_problem], s)) for s in SCOPES}
step_blocked, step_solved = (b - a for a, b in zip(rec["all cars"],
                                                   cost_and_benefit("all cars", 60)))
# Average cost per case at every threshold, and the cheapest price of any step past the
# recommended one: the two facts the "no elbow" sentence rests on.
avg = pd.DataFrame([cost_and_benefit("all cars", t) for t in GAPS[1:]],
                   index=GAPS[1:], columns=["blocked", "solved"])
avg_rises = bool((avg["blocked"] / avg["solved"]).is_monotonic_increasing)
steps = avg.diff().loc[RECOMMENDED + 30:]
cheapest_later_step = (steps["blocked"] / steps["solved"].where(steps["solved"] > 0)).min()

st.success(
    f"**{RECOMMENDED} minutes, on all cars.** Since the file records gaps in steps of 30 "
    f"minutes, this forbids exactly one thing: a rental starting the very minute the previous "
    f"one ends. It blocks {rec['all cars'][0]} ended rentals "
    f"({rec['all cars'][0] / is_ended.sum():.2%}) and solves {rec['all cars'][1]} of the "
    f"{rec_problems['all cars']} problematic cases "
    f"({rec['all cars'][1] / rec_problems['all cars']:.0%}), at "
    f"{rec['all cars'][0] / rec['all cars'][1]:.2f} rentals blocked per case — the cheapest step "
    f"on the curve. **There is no elbow beyond it**: the average cost per case "
    f"{'rises at every threshold' if avg_rises else 'does not fall below it'}, and every later "
    f"step costs at least {cheapest_later_step:.1f} rentals per extra case. Going to 60 minutes "
    f"would block {step_blocked} more rentals for {step_solved} more cases, "
    f"{step_blocked / step_solved:.1f} each. Whether that is worth it is a price the company "
    f"sets, not a result of this data."
)

st.info(
    f"**Why all cars and not one check-in type.** At {RECOMMENDED} minutes, Connect-only "
    f"costs {rec['Connect cars only'][0] / rec['Connect cars only'][1]:.2f} rentals blocked per "
    f"case solved, Mobile-only {rec['Mobile cars only'][0] / rec['Mobile cars only'][1]:.2f}, "
    f"all cars {rec['all cars'][0] / rec['all cars'][1]:.2f}. Connect drivers overlap the next "
    f"check-in less often (Q3), so Connect-only is the most expensive scope per case. "
    f"Mobile-only is the cheapest, but it leaves all {rec_problems['Connect cars only']} Connect "
    f"cases unsolved."
)

st.subheader("What this cannot tell you")
st.markdown(
    """
1. **A blocked rental is not a lost rental.** The file cannot say whether the driver booked
   another slot or another car, so the cost above is an upper bound.
2. **A solved case may not stay solved** if the blocked driver rebooks the same car at the
   first allowed slot and the previous driver is later than the buffer (Q4).
3. **The cancellations are a correlation.** The file records neither the reason nor the date
   of a cancellation (Q3).
4. **The file caps the gap at 12 hours**, so nothing beyond that is measurable.
5. **{blind} chained pairs have no recorded previous checkout** and are left out of Q3 and Q4
   rather than imputed.
""".format(blind=len(pairs) - len(measurable))
)

st.divider()
st.caption(
    f"Built from `get_around_delay_analysis.xlsx` — {len(delay):,} rentals.".replace(",", " ")
    + " The exploratory analysis is "
    "kept as an annex in `getaround_analysis.ipynb`. The pricing model asked for by the same "
    "brief is served by a separate API Space, documented at its `/docs`."
)
