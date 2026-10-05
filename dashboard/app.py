"""Getaround web app: two pages, one per audience.

  Delay analysis  -- for the product manager: the minimum delay between two rentals.
  Pricing         -- for a car owner: a suggested daily price, asked to the /predict API.

This file only declares the pages; each one lives in its own script. The sidebar controls of
the delay page are defined in delay.py, so they appear on that page only.
"""

import streamlit as st

st.set_page_config(page_title="Getaround", page_icon="🚗", layout="wide")

page = st.navigation([
    st.Page("delay.py", title="Delay analysis", icon="⏱️", default=True),
    st.Page("pricing.py", title="Pricing", icon="💶"),
])
page.run()
