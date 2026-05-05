import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone # Added timezone
import plotly.express as px
from sklearn.cluster import KMeans

# --- Page Config ---
st.set_page_config(page_title="SeismicWatch: Earthquake Analyzer", layout="wide")

# --- 1. Data Fetching (USGS API) ---
@st.cache_data(ttl=3600)
def load_earthquake_data(days=30):
    starttime = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={starttime}&minmagnitude=2.5"
    df = pd.read_csv(url)
    # Ensure time is datetime and UTC aware
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df

# --- Sidebar Filters ---
st.sidebar.header("📊 Global Filters")
days_to_look = st.sidebar.slider("Days of Data", 7, 90, 30)
df = load_earthquake_data(days_to_look)

min_m, max_m = float(df['mag'].min()), float(df['mag'].max())
mag_range = st.sidebar.slider("Magnitude Range", min_m, max_m, (4.0, max_m))

# Filter Data
filtered_df = df[(df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1])].copy()

# --- Main App ---
st.title("🌍 Earthquake Data Analysis & Risk Forecasting")
st.markdown(f"Displaying **{len(filtered_df)}** seismic events recorded in the last {days_to_look} days.")

# --- Section A: Interactive Map ---
st.header("📍 Interactive Seismic Map")
st.map(filtered_df[['latitude', 'longitude']])

# --- Section B & C: Visual Analysis ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Magnitude Distribution")
    fig_hist = px.histogram(filtered_df, x="mag", nbins=20, color_discrete_sequence=['#ff4b4b'])
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Depth vs Magnitude")
    fig_scatter = px.scatter(filtered_df, x="mag", y="depth", color="depth", 
                             labels={'depth': 'Depth (km)', 'mag': 'Magnitude'})
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Section D: Time Series Trends ---
st.header("⏱️ Time Series Trends")
# We use 'time' as index for resampling
ts_data = filtered_df.set_index('time').resample('D').count()['mag']
st.line_chart(ts_data)

# --- Section E: Risk Zone Classification (K-Means) ---
st.header("🚨 Risk Zone Classification")
cluster_data = filtered_df[['latitude', 'longitude', 'mag']].dropna()

if len(cluster_data) >= 3:
    kmeans = KMeans(n_clusters=3, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(cluster_data[['latitude', 'longitude']])
    
    # Sort clusters by average magnitude to assign risk labels
    means = cluster_data.groupby('cluster')['mag'].mean().sort_values()
    risk_labels = {means.index[0]: "Low Risk", means.index[1]: "Medium Risk", means.index[2]: "High Risk"}
    cluster_data['Risk Level'] = cluster_data['cluster'].map(risk_labels)

    fig_risk = px.scatter_mapbox(cluster_data, lat="latitude", lon="longitude", color="Risk Level",
                                 size="mag", hover_name="Risk Level",
                                 color_discrete_map={"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"},
                                 zoom=1, height=500)
    fig_risk.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_risk, use_container_width=True)
else:
    st.warning("Not enough data points for clustering. Try increasing the date range.")

# --- Section F: Probability Estimation ---
st.header("🤖 Magnitude Probability Estimation")

# FIXED LINE: Comparison using timezone.utc
now_utc = datetime.now(timezone.utc)
last_24h = filtered_df[filtered_df['time'] > (now_utc - timedelta(days=1))]

if not filtered_df.empty:
    recent_avg_mag = filtered_df['mag'].mean()
    event_count_24h = len(last_24h)
    
    # Heuristic probability calculation
    prob = (recent_avg_mag / 10) * (1 + (event_count_24h / 50))
    prob = min(max(prob, 0.05), 0.95) # Keep between 5% and 95% for realism

    st.metric(label="Likelihood of Significant Event (>5.0) in next 24h", value=f"{prob:.2%}")
    st.progress(prob)
else:
    st.write("Insufficient data for probability modeling.")
