---
title: getaround-delay-dashboard
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Getaround — Minimum delay between rentals

Should a car be hidden from search results when the requested check-in is too close to the previous
checkout? Two sliders — **how long** the minimum delay should be and **which cars** it applies to —
and every number on the page moves with them.

This page holds no model and calls no API. It is the delay analysis made interactive, recomputed
from the raw 21 310-rental file on every interaction, so no figure on it can be a stale constant
copied out of a notebook — which is also why it starts in seconds.

The same brief asks for a pricing model, and that one lives behind its own Space: the
[pricing API](https://lambla-getaround-pricing-api.hf.space), whose `/docs` is interactive.

## What it shows

1. **How many rentals the feature can reach at all.** Only 8.6% of rentals follow another rental of
   the same car; the feature cannot touch the rest.
2. **How often drivers are late, and whether it reaches the next driver.** The next rental's
   cancellation rate only moves once the overlap passes an hour.
3. **What each threshold and scope costs**, as one ratio: rentals blocked per problem avoided.
4. **The recommendation**, and the four things these files cannot answer.

## The finding

**The feature's entire target is 66 rentals out of 21 310** — the cases where the previous driver
came back more than an hour after the next check-in was due. Avoiding one costs between 10 and 25
blocked rentals, and the ratio only worsens as the threshold grows: there is no optimum on the
curve, only a price per avoided incident.

**"Connect cars only" is the wrong scope, for the opposite of the expected reason.** Connect drivers
are *less* late than mobile ones — 42.9% against 61.4%. Connect is where the problem shows up
because Connect cars are chained back-to-back three times more often, not because their drivers
behave worse.

## Known limitations

- **No revenue figure is possible.** The pricing dataset has no car identifier, so no rental has a
  price: every cost shown is a count of rentals standing in for euros.
- **A blocked rental is not a lost rental** — the file cannot say whether the driver rebooked.
- **The right-hand half of the impact chart rests on 103 pairs**, and a cancellation recorded after
  a late checkout is a correlation: the file records no cancellation reason.
- **The gap is capped at 12 hours** in the source file, so no longer threshold can be evaluated.

## Data

`data/get_around_delay_analysis.xlsx`, shipped inside the image — 751 KB, 21 310 rentals. There is
no database and no network call, which is why the page starts in seconds.
