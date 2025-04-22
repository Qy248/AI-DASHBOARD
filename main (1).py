import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

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
    .metric-box {
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
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

def create_cluster_insight(cluster_num, value_tier, recency, frequency, monetary, rfm_data):
    # Color mapping
    color_map = {
        'High Value': '#2ecc71',
        'Medium Value': '#f39c12',
        'Low Value': '#e74c3c'
    }
    tier_color = color_map.get(value_tier, '#3498db')
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'xy'}]],
        column_widths=[0.4, 0.6]
    )
    
    # Bullet chart for monetary value
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=monetary,
        number={'prefix': "$", 'font': {'size': 24}},
        title={'text': f"<b>Cluster {cluster_num}</b><br>{value_tier}", 'font': {'size': 18}},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, rfm_data['Monetary'].max()*1.1]},
            'bar': {'color': tier_color},
            'bgcolor': 'white',
            'borderwidth': 2
        }
    ), row=1, col=1)
    
    # RFM comparison bar chart
    fig.add_trace(go.Bar(
        x=['Recency (days)', 'Frequency', 'Monetary ($)'],
        y=[recency, frequency, monetary],
        name='Cluster Avg',
        marker_color=tier_color,
        text=[f"{recency:.1f}", f"{frequency:.1f}", f"${monetary:,.2f}"],
        textposition='auto'
    ), row=1, col=2)
    
    # Add reference lines using shapes instead of add_hline
    avg_recency = rfm_data['Recency'].mean()
    avg_frequency = rfm_data['Frequency'].mean()
    avg_monetary = rfm_data['Monetary'].mean()
    
    fig.add_shape(
        type="line",
        x0=-0.5, x1=2.5,
        y0=avg_recency, y1=avg_recency,
        line=dict(color="gray", dash="dot"),
        row=1, col=2
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5, x1=2.5,
        y0=avg_frequency, y1=avg_frequency,
        line=dict(color="gray", dash="dot"),
        row=1, col=2
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5, x1=2.5,
        y0=avg_monetary, y1=avg_monetary,
        line=dict(color="gray", dash="dot"),
        row=1, col=2
    )
    
    # Add annotation only for the first line
    fig.add_annotation(
        x=0, y=avg_recency*1.05,
        text="Dataset Average",
        showarrow=False,
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"<b>Cluster {cluster_num} Analysis</b><br>"
              f"<span style='color:{tier_color}'>{value_tier} Customers</span>",
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title_text="RFM Metrics", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    
    return fig

def show_cluster_analysis(rfm_data, algorithm, dimension):
    cluster_col = f"{algorithm}_Cluster_{dimension}"
    
    # Cluster distribution
    st.subheader("🔢 Cluster Distribution")
    cluster_counts = rfm_data[cluster_col].value_counts().sort_index()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clusters", len(cluster_counts))
    col2.metric("Total Customers", len(rfm_data))
    col3.metric("Avg Monetary Value", f"${rfm_data['Monetary'].mean():,.2f}")
    
    # Cluster visualization
    st.subheader("📊 Cluster Visualization")
    if dimension == "RF":
        fig = px.scatter(
            rfm_data,
            x="Recency",
            y="Frequency",
            color=cluster_col,
            title=f"{algorithm} Clustering (Recency vs Frequency)",
            hover_data=["Monetary"],
            color_continuous_scale=px.colors.sequential.Viridis
        )
    elif dimension == "FM":
        fig = px.scatter(
            rfm_data,
            x="Frequency",
            y="Monetary",
            color=cluster_col,
            title=f"{algorithm} Clustering (Frequency vs Monetary)",
            hover_data=["Recency"],
            color_continuous_scale=px.colors.sequential.Viridis
        )
    else:
        fig = px.scatter_3d(
            rfm_data,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color=cluster_col,
            title=f"{algorithm} Clustering (Recency vs Frequency vs Monetary)",
            color_continuous_scale=px.colors.sequential.Viridis
        )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster insights
    st.subheader("🔍 Detailed Cluster Insights")
    
    # Calculate cluster stats
    if dimension == "RF":
        cluster_stats = rfm_data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
    elif dimension == "FM":
        cluster_stats = rfm_data.groupby(cluster_col)[['Frequency', 'Monetary', 'Recency']].mean()
    else:
        cluster_stats = rfm_data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
    
    cluster_stats['Value Tier'] = cluster_stats['Monetary'].apply(
        lambda x: get_value_tier(x, rfm_data)[0]
    )
    
    # Sort clusters by monetary value
    sorted_clusters = cluster_stats.sort_values('Monetary', ascending=False).index
    
    for cluster in sorted_clusters:
        stats = cluster_stats.loc[cluster]
        tier, tier_class = get_value_tier(stats['Monetary'], rfm_data)
        
        with st.expander(f"Cluster {cluster} - {tier}", expanded=False):
            # Create insight visualization
            fig = create_cluster_insight(
                cluster, tier,
                stats['Recency'], stats['Frequency'], stats['Monetary'],
                rfm_data
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown(f"""
            <div class='cluster-card {tier_class}'>
                <h4>📌 Customer Characteristics</h4>
                <div class='metric-box'>
                    <b>Recency:</b> {stats['Recency']:.1f} days ({(stats['Recency']/rfm_data['Recency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Frequency:</b> {stats['Frequency']:.1f} purchases ({(stats['Frequency']/rfm_data['Frequency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Monetary:</b> ${stats['Monetary']:,.2f} ({(stats['Monetary']/rfm_data['Monetary'].mean()*100):.1f}% of average)
                </div>
                
                <h4>🎯 Recommended Actions</h4>
                {get_recommendations(tier, stats, rfm_data)}
            </div>
            """, unsafe_allow_html=True)

def get_recommendations(tier, stats, rfm_data):
    if tier == "High Value":
        return f"""
        <ol>
            <li><b>VIP Treatment:</b> Offer exclusive rewards for reaching ${stats['Monetary']*1.2:,.0f} quarterly spend</li>
            <li><b>Retention Focus:</b> Personal outreach every {max(14, int(stats['Recency']/2))} days</li>
            <li><b>Premium Upsell:</b> Recommend products 30-50% higher than current average spend</li>
            <li><b>Feedback:</b> Invite to customer advisory board</li>
        </ol>
        """
    elif tier == "Medium Value":
        return f"""
        <ol>
            <li><b>Upsell Strategy:</b> Bundle products to increase average order by 20-30%</li>
            <li><b>Engagement Boost:</b> Targeted emails every {max(7, int(stats['Recency']/3))} days</li>
            <li><b>Loyalty Program:</b> Offer points for reaching ${stats['Monetary']*1.5:,.0f} quarterly spend</li>
            <li><b>Feedback:</b> Survey to understand preferences</li>
        </ol>
        """
    else:
        return f"""
        <ol>
            <li><b>Win-Back Campaign:</b> Special discount after {int(stats['Recency']*1.2)} days inactivity</li>
            <li><b>Low-Cost Entry:</b> Offer starter products under ${rfm_data['Monetary'].quantile(0.25):.2f}</li>
            <li><b>Reactivation:</b> "We miss you" email after {int(stats['Recency']*1.5)} days</li>
            <li><b>Survey:</b> Understand barriers to purchasing</li>
        </ol>
        """

def main():
    st.title("📊 RFM Clustering Dashboard")
    st.markdown("""
    Customer segmentation using unsupervised learning (KMeans, BIRCH, GMM) based on RFM analysis.
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
