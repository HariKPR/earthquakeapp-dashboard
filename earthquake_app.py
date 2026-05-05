import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

# --- Defensive Import for Plotly ---
try:
    import plotly.express as px
except ImportError:
    st.error("Module 'plotly' not found. Please ensure 'plotly' is added to your requirements.txt file.")
    st.stop()

try:
    from sklearn.cluster import KMeans
except ImportError:
    st.error("Module 'scikit-learn' not found. Please ensure 'scikit-learn' is added to your requirements.txt file.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="SeismicWatch", layout="wide")

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def load_earthquake_data(days=30):
    try:
        starttime = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={starttime}&minmagnitude=2.5"
        df = pd.read_csv(url)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.header("📊 Settings")
days_to_look = st.sidebar.slider("Days of Data", 7, 90, 30)
raw_data = load_earthquake_data(days_to_look)

if not raw_data.empty:
    min_m = float(raw_data['mag'].min())
    max_m = float(raw_data['mag'].max())
    mag_range = st.sidebar.slider("Magnitude Range", min_m, max_m, (4.0, max_m))
    
    filtered_df = raw_data[(raw_data['mag'] >= mag_range[0]) & (raw_data['mag'].fillna(0) <= mag_range[1])].copy()

    # --- Main Dashboard ---
    st.title("🌍 Earthquake Analysis Dashboard")
    
    # Map Section
    st.subheader("📍 Global Event Locations")
    st.map(filtered_df[['latitude', 'longitude']])

    # Analysis Columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Magnitude Distribution**")
        fig1 = px.histogram(filtered_df, x="mag", color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.write("**Depth vs Magnitude**")
        fig2 = px.scatter(filtered_df, x="mag", y="depth", color="depth", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # Risk Zoning (K-Means)
    st.divider()
    st.subheader("🚨 Risk Zone Clustering (K-Means)")
    
    cluster_points = filtered_df[['latitude', 'longitude', 'mag']].dropna()
    if len(cluster_points) >= 3:
        kmeans = KMeans(n_clusters=3, n_init=10)
        cluster_points['cluster'] = kmeans.fit_predict(cluster_points[['latitude', 'longitude']])
        
        # Sort by magnitude to assign risk labels
        risk_order = cluster_points.groupby('cluster')['mag'].mean().sort_values().index
        risk_map = {risk_order[0]: "Low Risk", risk_order[1]: "Medium Risk", risk_order[2]: "High Risk"}
        cluster_points['Risk Level'] = cluster_points['cluster'].map(risk_map)
        
        fig_map = px.scatter_mapbox(cluster_points, lat="latitude", lon="longitude", color="Risk Level",
                                     size="mag", color_discrete_map={"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"},
                                     zoom=1, height=600, mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Increase the date/magnitude range to see Risk Clusters.")

else:
    st.warning("No data available for the selected filters.")
