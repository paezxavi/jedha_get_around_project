---
title: getaround-delay-dashboard
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Getaround — delay analysis and pricing

Two pages, one per audience.

## Delay analysis — for the product manager

Should a car be hidden from search results when a requested rental would start too soon after the
previous one ends? Two controls — **how long** the minimum delay is and **which cars** it applies
to — and every number on the page moves with them.

The page answers the product manager's four questions, one section each:

1. **Which share of the owners' revenue could the feature affect?** At most 8.93% of ended rentals.
2. **How many rentals would a threshold block?** A curve over every threshold, and the exact count
   at the chosen one.
3. **How often are drivers late for the next check-in, and what does it do to the next driver?**
   12.6% of chained pairs overlap; long overlaps go with far more cancellations.
4. **How many problematic cases would a threshold solve?** The same curve, on the benefit side.

It then puts cost against benefit, threshold by threshold, for the chosen scope.

Every number is recomputed from the raw 21 310-rental file on each interaction; the counting rules
are written on the page next to the numbers they shape. This page holds no model.

## Pricing — for a car owner

Describe a car and the page suggests a daily rental price. It holds no model either: it sends the
car to the [pricing API](https://lambla-getaround-pricing-api.hf.space) `/predict`, shows the
answer, and shows the request and the response it exchanged. The column order is read from the
API's `/health`, and the dropdowns offer only values the model was trained on.

## Configuration

One optional variable, `PRICING_API_URL`, points the Pricing page at another API — a local one
during development. It defaults to the online Space. No secret: the delay page reads files shipped
in the image, and the API is public.

## Data

`data/get_around_delay_analysis.xlsx` (21 310 rentals) for the delay page and
`data/get_around_pricing_project.csv` (4 843 cars) for the Pricing page's dropdowns, both shipped
inside the image.
