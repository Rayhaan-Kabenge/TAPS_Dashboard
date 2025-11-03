import streamlit as st

# Page config
st.set_page_config(
    page_title="KSUTAPS Decision Support Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple custom CSS for button styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button hover effect */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        border-color: #5B2A86 !important;
        color: #5B2A86 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(91, 42, 134, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌾 KSUTAPS Decision Support Suite")
st.markdown("### Weather, Crop Health, and Soil insights—unified for faster, better decisions")
st.divider()

# Welcome text
st.markdown("""
Welcome to the **KSUTAPS Decision Support Suite**—your unified entry to weather, crop health, and soil insights. 
Make data-driven decisions with confidence using real-time monitoring and analytics.
""")

st.divider()

# Dashboard cards
st.markdown("## Select a Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ☁️ Weather Dashboard")
    st.markdown("""
    Real-time weather, reference ET, and short-term forecast summaries to guide irrigation decisions.
    """)
    if st.button("Open Weather Dashboard", key="weather", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Weather_Dashboard.py")

with col2:
    st.markdown("### 🌱 Crop Health Dashboard")
    st.markdown("""
    Compare NDVI/MCARI2 across plots and dates, with nitrogen/irrigation overlays for context.
    """)
    if st.button("Open Crop Health Dashboard", key="crop", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Crop_Health_Dashboard.py")

with col3:
    st.markdown("### 🌍 Soil Dashboard")
    st.info("**Coming Soon**")
    st.markdown("""
    Root-zone moisture, EC, and temperature monitoring—coming soon.
    """)
    if st.button("Preview Soil Dashboard", key="soil", use_container_width=True):
        st.switch_page("pages/3_Soil_Dashboard_Coming_Soon.py")

st.divider()

# Email notification for Soil
with st.expander("📧 Get notified when Soil Dashboard launches"):
    email = st.text_input("Enter your email address", placeholder="your.email@example.com")
    if st.button("Notify Me", type="primary"):
        if email:
            st.success("✅ Thanks! We'll notify you when the Soil Dashboard launches.")
        else:
            st.warning("Please enter a valid email address.")

# Footer
st.divider()
col_left, col_mid, col_right = st.columns(3)
with col_left:
    st.caption("© 2025 KSUTAPS • Kansas State University")
with col_mid:
    st.caption("[About](#) • [Data Sources](#) • [Credits](#)")
with col_right:
    st.caption("Dashboard • v1.0")





#. streamlit run "/Users/rayhaankabenge/Desktop/KSUTAPS/2025/Dashboard_v2/v3/home.py"    