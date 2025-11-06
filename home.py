import streamlit as st
from datetime import datetime

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="KSUTAPS Decision Support Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------
# Global styles (typography, spacing, cards, buttons)
# ------------------------------
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Type scale & rhythm */
:root{
  --brand:#5B2A86;         /* KSU-adjacent accent */
  --text:#1f2328;
  --muted:#6b7280;
  --bg:#0f1116;            /* dark header bg if needed later */
  --card-radius:18px;
  --shadow:0 2px 10px rgba(0,0,0,.06);
  --shadow-lg:0 10px 30px rgba(0,0,0,.12);
}
h1, h2, h3 { letter-spacing:.2px; }
p, .markdown-text { line-height:1.45; }

/* Section spacing */
.block { margin-top: .25rem; margin-bottom: .25rem; }
.section { margin-top: 0.25rem; margin-bottom: 1.25rem; }

/* Grid card */
.ksu-card{
  position: relative;
  border:1px solid rgba(0,0,0,.06);
  border-radius: var(--card-radius);
  padding: 1.1rem 1.2rem;
  background: #ffffff;
  box-shadow: var(--shadow);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  height: 100%;
}
.ksu-card:hover{
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
  border-color: rgba(91,42,134,.25);
}
.ksu-card h3 { margin-top: 0; margin-bottom: .35rem; }
.ksu-card p  { margin: .25rem 0 .9rem 0; color: var(--muted); }

/* Subtle status chip (optional, not used on soil per spec) */
.chip{
  display:inline-block; font-size:.78rem; font-weight:600;
  padding:.18rem .5rem; border-radius:999px; color:#fff; background:var(--brand);
}

/* Primary buttons: hover focus */
.stButton > button {
  border: 1px solid rgba(91, 42, 134, .35) !important;
  border-radius: 12px !important;
  transition: all .18s ease !important;
  font-weight: 600 !important;
}
.stButton > button:hover {
  border-color: var(--brand) !important;
  color: var(--brand) !important;
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(91,42,134,.20);
}

/* Disabled button visual consistency */
.stButton > button:disabled {
  opacity:.65 !important;
  cursor:not-allowed !important;
}

/* Footer links */
.footer a { color: var(--muted); text-decoration: none; }
.footer a:hover { color: var(--brand); text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header: logo + title block
# ------------------------------
col_logo, col_title = st.columns([0.14, 1], gap="small")
with col_logo:
    # Swap with your actual logo path if different
    st.image("Files/ksu_logo.png", caption=None, width=520)
with col_title:
    st.title("KSUTAPS Decision Support Suite")
    st.markdown("Weather, Crop Health, and Soil insights—unified for faster, better decisions.")

st.divider()

# ------------------------------
# Welcome / intro
# ------------------------------
st.markdown(
    "Welcome to the **KSUTAPS Decision Support Suite**—your unified entry to weather, crop health, and soil insights. "
    "Make data-driven decisions with confidence using real-time monitoring and analytics."
)
st.divider()

# ------------------------------
# Dashboard selection
# ------------------------------
st.subheader("Select a Dashboard")

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="ksu-card">', unsafe_allow_html=True)
    st.markdown("### ☁️ Weather Module")
    st.markdown("Real-time weather, reference ET, and short-term forecast summaries to guide irrigation decisions.")
    if st.button("Open Weather Dashboard", key="weather", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Weather_Dashboard.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="ksu-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Crop Module")
    st.markdown("Compare NDVI/MCARI2 across plots and dates, with nitrogen/irrigation overlays for context.")
    if st.button("Open Crop Health Dashboard", key="crop", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Crop_Health_Dashboard.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="ksu-card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Soil Module")
    st.markdown("_Root-zone moisture, EC, and temperature monitoring — coming soon._")
    st.button("Preview Soil Dashboard", key="soil", use_container_width=True, disabled=True, help="This module is in development.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ------------------------------
# Notify when Soil launches
# ------------------------------
with st.expander("📧 Get notified when Soil Dashboard launches"):
    left, right = st.columns([1, .35])
    with left:
        email = st.text_input("Email address", placeholder="your.email@example.com")
    with right:
        notify_click = st.button("Notify Me", type="primary", use_container_width=True)
    if notify_click:
        if email and "@" in email and "." in email:
            # Placeholder success; wire up to your backend/email service if needed
            st.success("✅ Thanks! We'll notify you when the Soil Dashboard launches.")
        else:
            st.warning("Please enter a valid email address.")

# ------------------------------
# Footer
# ------------------------------
st.divider()
f1, f2, f3 = st.columns(3)
with f1:
    st.caption("© 2025 KSUTAPS • Kansas State University")
with f2:
    st.caption('<span class="footer">\
        <a href="#">About</a> • <a href="#">Data Sources</a> • <a href="#">Credits</a>\
    </span>', unsafe_allow_html=True)
with f3:
    st.caption(f"Dashboard • v1.0 • {datetime.now().strftime('%Y-%m-%d')}")

# . streamlit run /path/to/home.py
