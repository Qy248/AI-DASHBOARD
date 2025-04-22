import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page config
st.set_page_config(
    page_title='Customer Segmentation',
    page_icon=':bar_chart:',
    layout='wide'
)

# --- Load data and models ---
@st.cache_resource
def load_data_models():
    # Models
    models = {
        "KMeans": {
            "RF": joblib.load("final_kmeans_RF_model.joblib"),
            "FM": joblib.load("final_kmeans_FM_model.joblib"),
            "RFM": joblib.load("final_kmeans_RFM_model.joblib"),
        },
        "GMM": {
            "RF": joblib.load("GMM_RF_model.joblib"),
            "FM": joblib.load("GMM_FM_model.joblib"),
            "RFM": joblib.load("GMM_RFM_model.joblib"),
        },
        "BIRCH": {
            "RF": joblib.load("BIRCH_RF_model.joblib"),
            "FM": joblib.load("BIRCH_FM_model.joblib"),
            "RFM": joblib.load("BIRCH_RFM_model.joblib"),
        }
    }

    # DataFrames
    dataframes = {
        "KMeans": {
            "RF": joblib.load("rfm_data_RF_with_cluster.joblib"),
            "FM": joblib.load("rfm_data_FM_with_cluster.joblib"),
            "RFM": joblib.load("rfm_data_RFM_with_cluster.joblib"),
        },
        "GMM": {
            "RF": joblib.load("rfm_data_GMM_RF_clusters.joblib"),
            "FM": joblib.load("rfm_data_GMM_FM_clusters.joblib"),
            "RFM": joblib.load("rfm_data_GMM_RFM_clusters.joblib"),
        },
        "BIRCH": {
            "RF": joblib.load("rfm_data_RF_clusters.joblib"),
            "FM": joblib.load("rfm_data_FM_clusters.joblib"),
            "RFM": joblib.load("rfm_data_RFM_clusters.joblib"),
        }
    }

    # Scaler
    scaler = joblib.load("rfm_scaler.joblib")
    return models, dataframes, scaler

# --- Plotting ---
def plot_clusters(df, cluster_col, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette="Set2", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# --- Main app ---
def main():
    st.title("Customer Segmentation Dashboard")

    # Load data and models
    kmeans_df, gmm_df, birch_df, kmeans_model, gmm_model, birch_model, scaler = load_all_data()

    # Sidebar model selector
    st.sidebar.header("Select a Clustering Model")
    model_choice = st.sidebar.selectbox("Model", ["KMeans", "GMM", "BIRCH"])

    # Get data based on model selection
    if model_choice == "KMeans":
        df = kmeans_df.copy()
        model = kmeans_model
        cluster_col = "KMeans_Cluster"
    elif model_choice == "GMM":
        df = gmm_df.copy()
        model = gmm_model
        cluster_col = "GMM_Cluster"
    else:
        df = birch_df.copy()
        model = birch_model
        cluster_col = "BIRCH_Cluster"

    # Sample Data
    st.markdown(f"### Sample Data ({model_choice})")
    st.dataframe(df.head())

    today = datetime(2023, 9, 15)

    # Recency Input
    st.markdown("### Most Recent Purchase Date")
    recency_slider = st.slider(
        "Select your most recent purchase date",
        min_value=today - timedelta(days=365),
        max_value=today,
        value=today - timedelta(days=30),
        format="YYYY-MM-DD"
    )
    recency_days = (today - recency_slider).days
    st.write(f"Recency: {recency_days} days")

    # Frequency Input
    st.markdown("### Frequency of Visiting the Store")
    frequency_input = st.slider("Number of Visits", min_value=1, max_value=380, value=10)
    st.write(f"Frequency: {frequency_input}")

    # Monetary Input
    st.markdown("### Total Amount Spent")
    monetary_input = st.slider("Total Spent", min_value=1.0, max_value=300000.0, value=100.0, step=10.0)
    st.write(f"Monetary: {monetary_input}")

    # Predict cluster
    user_data = np.array([[recency_days, frequency_input, monetary_input]])
    user_scaled = scaler.transform(user_data)

    try:
        prediction = model.predict(user_scaled)[0]
        st.success(f"Predicted Cluster: {prediction} ({model_choice})")
    except Exception as e:
        st.error(f"Prediction error: {e}")

    # Main cluster plots
    st.markdown(f"### Recency vs Frequency Clustering ({model_choice})")
    plot_cluster(df, cluster_col, 'Recency', 'Frequency', model_choice)

    st.markdown(f"### Frequency vs Monetary Clustering ({model_choice})")
    plot_cluster(df, cluster_col, 'Frequency', 'Monetary', model_choice)

    st.markdown(f"### Recency vs Monetary Clustering ({model_choice})")
    plot_cluster(df, cluster_col, 'Recency', 'Monetary', model_choice)

    # 🔽 Additional Graphs 🔽

    # Cluster count bar plot
    st.markdown("### Cluster Count Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x=cluster_col, palette="Set2", ax=ax1)
    ax1.set_title("Number of Customers per Cluster")
    st.pyplot(fig1)

    # Boxplots
    st.markdown("### Boxplots of RFM by Cluster")
    for feature in ['Recency', 'Frequency', 'Monetary']:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df, x=cluster_col, y=feature, palette="Set3", ax=ax_box)
        ax_box.set_title(f"{feature} by Cluster")
        st.pyplot(fig_box)

    # Pairplot (optional: slow for large data)
    st.markdown("### RFM Pairplot by Cluster")
    st.info("This may take a few seconds for large datasets.")
    sample_df = df[[cluster_col, 'Recency', 'Frequency', 'Monetary']].sample(n=500, random_state=42) if len(df) > 500 else df
    fig_pair = sns.pairplot(sample_df, hue=cluster_col, palette="husl")
    st.pyplot(fig_pair)

    # Optional: 3D plot (if using Plotly)
    st.markdown("### 3D Cluster Plot (Recency, Frequency, Monetary)")
    try:
        import plotly.express as px
        fig3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary',
                              color=cluster_col, title="3D RFM Cluster View")
        st.plotly_chart(fig3d)
    except:
        st.warning("Plotly not installed. Run `pip install plotly` if you'd like to enable the 3D view.")


if __name__ == "__main__":
    main()
