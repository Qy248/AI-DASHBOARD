import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="RFM Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_data
def load_data():
    rfm_data = joblib.load("rfm_data_original.joblib")
    models = {
        'KMeans': {
            'RF': joblib.load("KMeans_RF_model.joblib"),
            'FM': joblib.load("KMeans_FM_model.joblib"),
            'RFM': joblib.load("KMeans_RFM_model.joblib")
        },
        'BIRCH': {
            'RF': joblib.load("BIRCH_RF_model.joblib"),
            'FM': joblib.load("BIRCH_FM_model.joblib"),
            'RFM': joblib.load("BIRCH_RFM_model.joblib")
        },
        'GMM': {
            'RF': joblib.load("GMM_RF_model.joblib"),
            'FM': joblib.load("GMM_FM_model.joblib"),
            'RFM': joblib.load("GMM_RFM_model.joblib")
        }
    }
    return rfm_data, models

rfm_data, models = load_data()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
algorithm = st.sidebar.selectbox("Select Algorithm", ["KMeans", "BIRCH", "GMM"])
dimension = st.sidebar.selectbox("Select Dimension", ["RF", "FM", "RFM"])
cluster_to_analyze = st.sidebar.selectbox("Select Cluster to Analyze", sorted(rfm_data[f"{algorithm}_Cluster_{dimension}"].unique()))

# Main dashboard
st.title("📊 RFM Clustering Analysis Dashboard")
st.markdown("""
This dashboard visualizes customer segmentation using RFM (Recency, Frequency, Monetary) analysis 
with different clustering algorithms.
""")

# Metrics row
st.subheader("📈 Overall Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(rfm_data))
col2.metric(f"Clusters in {algorithm}-{dimension}", rfm_data[f"{algorithm}_Cluster_{dimension}"].nunique())
col3.metric("Avg. Monetary Value", f"${rfm_data['Monetary'].mean():,.2f}")

# Cluster distribution
st.subheader("🔢 Cluster Distribution")
fig1 = px.pie(
    rfm_data, 
    names=f"{algorithm}_Cluster_{dimension}",
    title=f"Cluster Distribution ({algorithm}-{dimension})"
)
st.plotly_chart(fig1, use_container_width=True)

# RFM plots
st.subheader("📊 RFM Cluster Visualization")

if dimension == "RF":
    fig2 = px.scatter(
        rfm_data,
        x="Recency",
        y="Frequency",
        color=f"{algorithm}_Cluster_{dimension}",
        title=f"{algorithm} Clustering (Recency vs Frequency)",
        hover_data=["Monetary"]
    )
elif dimension == "FM":
    fig2 = px.scatter(
        rfm_data,
        x="Frequency",
        y="Monetary",
        color=f"{algorithm}_Cluster_{dimension}",
        title=f"{algorithm} Clustering (Frequency vs Monetary)",
        hover_data=["Recency"]
    )
else:  # RFM
    fig2 = px.scatter_3d(
        rfm_data,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color=f"{algorithm}_Cluster_{dimension}",
        title=f"{algorithm} Clustering (Recency vs Frequency vs Monetary)"
    )
st.plotly_chart(fig2, use_container_width=True)

# Cluster analysis
st.subheader(f"🔍 Cluster {cluster_to_analyze} Detailed Analysis")

cluster_data = rfm_data[rfm_data[f"{algorithm}_Cluster_{dimension}"] == cluster_to_analyze]
non_cluster_data = rfm_data[rfm_data[f"{algorithm}_Cluster_{dimension}"] != cluster_to_analyze]

col1, col2, col3 = st.columns(3)
col1.metric("Customers in Cluster", len(cluster_data))
col2.metric("Avg Recency", f"{cluster_data['Recency'].mean():.1f} days")
col3.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Monetary", f"${cluster_data['Monetary'].mean():,.2f}")
col2.metric("% of Total Revenue", f"{(cluster_data['Monetary'].sum() / rfm_data['Monetary'].sum())*100:.1f}%")
col3.metric("Avg Customer Value", f"${cluster_data['Monetary'].mean() / cluster_data['Frequency'].mean():,.2f} per purchase")

# Comparison plots
st.subheader("📉 Cluster Comparison")

tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])

with tab1:
    fig_rec = go.Figure()
    fig_rec.add_trace(go.Box(y=cluster_data['Recency'], name=f'Cluster {cluster_to_analyze}'))
    fig_rec.add_trace(go.Box(y=non_cluster_data['Recency'], name='Other Clusters'))
    fig_rec.update_layout(title="Recency Comparison")
    st.plotly_chart(fig_rec, use_container_width=True)

with tab2:
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Box(y=cluster_data['Frequency'], name=f'Cluster {cluster_to_analyze}'))
    fig_freq.add_trace(go.Box(y=non_cluster_data['Frequency'], name='Other Clusters'))
    fig_freq.update_layout(title="Frequency Comparison")
    st.plotly_chart(fig_freq, use_container_width=True)

with tab3:
    fig_mon = go.Figure()
    fig_mon.add_trace(go.Box(y=cluster_data['Monetary'], name=f'Cluster {cluster_to_analyze}'))
    fig_mon.add_trace(go.Box(y=non_cluster_data['Monetary'], name='Other Clusters'))
    fig_mon.update_layout(title="Monetary Comparison")
    st.plotly_chart(fig_mon, use_container_width=True)

# Cluster characteristics
st.subheader("📋 Cluster Characteristics")

if dimension == "RF":
    cluster_means = rfm_data.groupby(f"{algorithm}_Cluster_{dimension}")[['Recency', 'Frequency']].mean()
elif dimension == "FM":
    cluster_means = rfm_data.groupby(f"{algorithm}_Cluster_{dimension}")[['Frequency', 'Monetary']].mean()
else:
    cluster_means = rfm_data.groupby(f"{algorithm}_Cluster_{dimension}")[['Recency', 'Frequency', 'Monetary']].mean()

st.dataframe(cluster_means.style.background_gradient(cmap='Blues'), use_container_width=True)

# Recommendations
st.subheader("💡 Marketing Recommendations")

if dimension == "RF":
    if cluster_data['Recency'].mean() < rfm_data['Recency'].mean() and cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean():
        st.success("**Best Customers**: Recent and frequent purchasers. Reward them to encourage loyalty.")
    elif cluster_data['Recency'].mean() > rfm_data['Recency'].mean() and cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean():
        st.warning("**At Risk Customers**: Used to purchase often but haven't recently. Win them back with reactivation campaigns.")
    elif cluster_data['Recency'].mean() < rfm_data['Recency'].mean() and cluster_data['Frequency'].mean() < rfm_data['Frequency'].mean():
        st.info("**New Customers**: Purchased recently but not often. Create onboarding to increase frequency.")
    else:
        st.error("**Hibernating Customers**: Not purchased recently or often. Consider low-cost reactivation or phase out.")

elif dimension == "FM":
    if cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean() and cluster_data['Monetary'].mean() > rfm_data['Monetary'].mean():
        st.success("**High Value Customers**: Frequent and high spending. Offer exclusive benefits and premium services.")
    elif cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean() and cluster_data['Monetary'].mean() < rfm_data['Monetary'].mean():
        st.info("**Budget Shoppers**: Buy often but spend little. Cross-sell higher value items.")
    elif cluster_data['Frequency'].mean() < rfm_data['Frequency'].mean() and cluster_data['Monetary'].mean() > rfm_data['Monetary'].mean():
        st.warning("**Big Spenders**: Infrequent but high value purchases. Target with high-end offers.")
    else:
        st.error("**Low Engagement**: Infrequent and low spending. Consider win-back campaigns or budget offers.")

else:  # RFM
    if (cluster_data['Recency'].mean() < rfm_data['Recency'].mean() and 
        cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean() and 
        cluster_data['Monetary'].mean() > rfm_data['Monetary'].mean()):
        st.success("**Champions**: Recent, frequent and high value. Nurture with VIP treatment.")
    elif (cluster_data['Recency'].mean() > rfm_data['Recency'].mean() and 
          cluster_data['Frequency'].mean() > rfm_data['Frequency'].mean() and 
          cluster_data['Monetary'].mean() > rfm_data['Monetary'].mean()):
        st.warning("**Can't Lose Them**: Used to be high value but haven't purchased recently. Win them back urgently.")
    elif (cluster_data['Recency'].mean() < rfm_data['Recency'].mean() and 
          cluster_data['Frequency'].mean() < rfm_data['Frequency'].mean() and 
          cluster_data['Monetary'].mean() < rfm_data['Monetary'].mean()):
        st.info("**New Customers**: Need onboarding to increase frequency and value.")
    else:
        st.error("**Need Attention**: Mixed behavior. Requires targeted analysis and campaigns.")

# Data table
st.subheader("📁 Customer Data")
st.dataframe(rfm_data.sort_values(by='Monetary', ascending=False), use_container_width=True)

# Download button
st.sidebar.download_button(
    label="Download Cluster Data",
    data=rfm_data.to_csv().encode('utf-8'),
    file_name='rfm_cluster_data.csv',
    mime='text/csv'
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**RFM Clustering Dashboard**  
Created with Streamlit  
Data last updated: {}
""".format(pd.to_datetime('today').strftime('%Y-%m-%d')))
