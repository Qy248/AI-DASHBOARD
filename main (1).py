import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
import plotly.express as px
import plotly.graph_objects as go

def main():
    # Set page config
    st.set_page_config(
        page_title="RFM Clustering Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .cluster-card {
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .high-value {
            background-color: #e6f7e6;
            border-left: 5px solid #4CAF50;
        }
        .medium-value {
            background-color: #fff3e6;
            border-left: 5px solid #FF9800;
        }
        .low-value {
            background-color: #ffebee;
            border-left: 5px solid #F44336;
        }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        try:
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
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None

    def get_value_tier(monetary_value, rfm_data):
        q25 = rfm_data['Monetary'].quantile(0.25)
        q75 = rfm_data['Monetary'].quantile(0.75)
        
        if monetary_value > q75:
            return "High Value", "high-value"
        elif monetary_value > q25:
            return "Medium Value", "medium-value"
        else:
            return "Low Value", "low-value"

    def show_cluster_analysis(rfm_data, algorithm, dimension):
        cluster_col = f"{algorithm}_Cluster_{dimension}"
        
        # Cluster distribution
        st.subheader("🔢 Cluster Distribution")
        cluster_counts = rfm_data[cluster_col].value_counts().sort_index()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clusters", len(cluster_counts))
        col2.metric("Total Customers", len(rfm_data))
        col3.metric("Avg Monetary Value", f"${rfm_data['Monetary'].mean():,.2f}")

        fig1 = px.pie(
            cluster_counts,
            values=cluster_counts.values,
            names=cluster_counts.index,
            title=f"Customer Distribution Across Clusters ({algorithm}-{dimension})",
            hole=0.3
        )
        st.plotly_chart(fig1, use_container_width=True)

        # RFM visualization
        st.subheader("📊 Cluster Visualization")
        if dimension == "RF":
            fig2 = px.scatter(
                rfm_data,
                x="Recency",
                y="Frequency",
                color=cluster_col,
                title=f"{algorithm} Clustering (Recency vs Frequency)",
                hover_data=["Monetary"],
                color_continuous_scale=px.colors.sequential.Viridis
            )
        elif dimension == "FM":
            fig2 = px.scatter(
                rfm_data,
                x="Frequency",
                y="Monetary",
                color=cluster_col,
                title=f"{algorithm} Clustering (Frequency vs Monetary)",
                hover_data=["Recency"],
                color_continuous_scale=px.colors.sequential.Viridis
            )
        else:  # RFM
            fig2 = px.scatter_3d(
                rfm_data,
                x="Recency",
                y="Frequency",
                z="Monetary",
                color=cluster_col,
                title=f"{algorithm} Clustering (Recency vs Frequency vs Monetary)",
                color_continuous_scale=px.colors.sequential.Viridis
            )
        st.plotly_chart(fig2, use_container_width=True)

        # Cluster statistics
        st.subheader("📋 Cluster Statistics")
        if dimension == "RF":
            cluster_stats = rfm_data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
        elif dimension == "FM":
            cluster_stats = rfm_data.groupby(cluster_col)[['Frequency', 'Monetary', 'Recency']].mean()
        else:
            cluster_stats = rfm_data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()

        cluster_stats['Value Tier'] = cluster_stats['Monetary'].apply(
            lambda x: get_value_tier(x, rfm_data)[0]
        )

        st.dataframe(
            cluster_stats.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

        # Cluster insights
        st.subheader("🔍 Cluster Insights")
        sorted_clusters = cluster_stats.sort_values('Monetary', ascending=False).index

        for cluster in sorted_clusters:
            stats = cluster_stats.loc[cluster]
            tier, tier_class = get_value_tier(stats['Monetary'], rfm_data)
            
            with st.expander(f"**Cluster {cluster}** ({tier})", expanded=False):
                st.markdown(
                    f"""
                    <div class="cluster-card {tier_class}">
                        <h4>📌 Cluster {cluster} - {tier}</h4>
                        <p><b>Avg Recency:</b> {stats['Recency']:.1f} days</p>
                        <p><b>Avg Frequency:</b> {stats['Frequency']:.1f} purchases</p>
                        <p><b>Avg Monetary:</b> ${stats['Monetary']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("**🎯 Recommended Actions:**")
                if tier == "High Value":
                    st.markdown("""
                    - **VIP Treatment**: Exclusive rewards, early access
                    - **Personalization**: Tailored recommendations
                    - **Retention Focus**: Loyalty programs, dedicated support
                    """)
                elif tier == "Medium Value":
                    st.markdown("""
                    - **Upsell Opportunities**: Complementary products
                    - **Engagement Boost**: Email campaigns, limited offers
                    - **Feedback Collection**: Understand preferences
                    """)
                else:
                    st.markdown("""
                    - **Win-Back Campaigns**: Special discounts
                    - **Low-Cost Incentives**: Free shipping
                    - **Reactivation Strategies**: Survey inactive customers
                    """)

    # Main app
    st.title("📊 RFM Clustering Dashboard (Unsupervised Learning)")
    st.markdown("""
    Visualizing customer segments using **unsupervised clustering** (KMeans, BIRCH, GMM) based on RFM analysis.
    """)

    # Load data
    rfm_data, models = load_data()
    if rfm_data is None or models is None:
        st.stop()

    # Sidebar controls
    st.sidebar.header("Clustering Configuration")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["KMeans", "BIRCH", "GMM"],
        key='algorithm'
    )
    dimension = st.sidebar.selectbox(
        "Select Dimensions",
        ["RF", "FM", "RFM"],
        key='dimension'
    )

    # Show analysis
    show_cluster_analysis(rfm_data, algorithm, dimension)

    # Data export
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Data")
    if st.sidebar.button("Download Cluster Data"):
        csv = rfm_data.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download as CSV",
            data=csv,
            file_name="rfm_clusters.csv",
            mime="text/csv"
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **RFM Clustering Dashboard**  
    *Unsupervised Learning Approach*  
    Data last updated: {}
    """.format(pd.to_datetime('today').strftime('%Y-%m-%d')))

if __name__ == "__main__":
    main()
