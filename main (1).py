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

# Custom CSS for blue theme with black text
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .cluster-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #e6f2ff;
        border-left: 5px solid #1a73e8;
    }
    .metric-box {
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f7ff;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    .st-b7 {
        color: #000000 !important;
    }
    .st-bb {
        background-color: #1a73e8 !important;
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
        return "High Value", "#1a73e8"  # Dark blue
    elif monetary_value > q25:
        return "Medium Value", "#4285f4"  # Medium blue
    else:
        return "Low Value", "#8ab4f8"  # Light blue

def create_cluster_analysis(cluster_num, stats, rfm_data):
    tier, tier_color = get_value_tier(stats['Monetary'], rfm_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        subplot_titles=(
            "Monetary Value", 
            "Cluster Distribution", 
            "RFM Comparison", 
            "Recency vs Frequency"
        )
    )
    
    # Monetary value indicator
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=stats['Monetary'],
        number={'prefix': "$", 'font': {'size': 24, 'color': 'black'}},
        title={'text': f"<b>Cluster {cluster_num}</b><br>{tier}", 'font': {'color': 'black'}},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, rfm_data['Monetary'].max()*1.1], 'tickcolor': 'black'},
            'bar': {'color': tier_color},
            'bgcolor': 'white',
            'borderwidth': 2
        }
    ), row=1, col=1)
    
    # Cluster distribution pie chart
    cluster_counts = rfm_data['Cluster'].value_counts()
    fig.add_trace(go.Pie(
        labels=cluster_counts.index,
        values=cluster_counts.values,
        marker=dict(colors=['#1a73e8', '#4285f4', '#8ab4f8']),
        row=1, col=2
    )
    
    # RFM comparison bar chart
    fig.add_trace(go.Bar(
        x=['Recency', 'Frequency', 'Monetary'],
        y=[stats['Recency'], stats['Frequency'], stats['Monetary']],
        marker_color=tier_color,
        text=[f"{stats['Recency']:.1f}", f"{stats['Frequency']:.1f}", f"${stats['Monetary']:,.2f}"],
        textposition='auto',
        textfont={'color': 'black'}
    ), row=2, col=1)
    
    # Recency vs Frequency scatter
    cluster_data = rfm_data[rfm_data['Cluster'] == cluster_num]
    fig.add_trace(go.Scatter(
        x=cluster_data['Recency'],
        y=cluster_data['Frequency'],
        mode='markers',
        marker=dict(color=tier_color),
        name=f'Cluster {cluster_num}'
    ), row=2, col=2)
    
    # Add reference lines
    fig.add_hline(y=rfm_data['Recency'].mean(), line_dash="dot", row=2, col=2)
    fig.add_vline(x=rfm_data['Frequency'].mean(), line_dash="dot", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title=f"<b>Cluster {cluster_num} Comprehensive Analysis</b>",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    
    return fig, tier, tier_color

def main():
    st.title("📊 RFM Clustering Dashboard")
    st.markdown("""
    <style>
        div[data-testid="stMarkdownContainer"] p {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    rfm_data, models = load_data()
    if rfm_data is None or models is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["KMeans", "BIRCH", "GMM"],
        key='algorithm'
    )
    dimension = st.sidebar.selectbox(
        "Dimensions",
        ["RF", "FM", "RFM"],
        key='dimension'
    )
    
    # Cluster distribution
    st.header("Cluster Distribution")
    cluster_col = f"{algorithm}_Cluster_{dimension}"
    cluster_counts = rfm_data[cluster_col].value_counts().sort_index()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clusters", len(cluster_counts))
    col2.metric("Total Customers", len(rfm_data))
    col3.metric("Avg Monetary", f"${rfm_data['Monetary'].mean():,.2f}")
    
    # Cluster pie chart
    fig_pie = px.pie(
        cluster_counts,
        values=cluster_counts.values,
        names=cluster_counts.index,
        color=cluster_counts.index,
        color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'],
        hole=0.3
    )
    fig_pie.update_layout(
        title="Customer Distribution Across Clusters",
        font=dict(color='black')
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Cluster statistics
    cluster_stats = rfm_data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_stats['Size'] = rfm_data.groupby(cluster_col).size()
    
    # Detailed cluster analysis
    st.header("Cluster Deep Dive Analysis")
    for cluster in cluster_stats.index:
        stats = cluster_stats.loc[cluster]
        fig, tier, tier_color = create_cluster_analysis(cluster, stats, rfm_data)
        
        with st.expander(f"Cluster {cluster} - {tier}", expanded=False):
            st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown(f"""
            <div class='cluster-card'>
                <h4 style='color: black;'>📊 Cluster {cluster} Characteristics</h4>
                <div class='metric-box'>
                    <b>Recency:</b> {stats['Recency']:.1f} days ({(stats['Recency']/rfm_data['Recency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Frequency:</b> {stats['Frequency']:.1f} purchases ({(stats['Frequency']/rfm_data['Frequency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Monetary:</b> ${stats['Monetary']:,.2f} ({(stats['Monetary']/rfm_data['Monetary'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Customers:</b> {stats['Size']} ({(stats['Size']/len(rfm_data)*100:.1f}% of total)
                </div>
                
                <h4 style='color: black;'>🎯 Recommended Actions</h4>
                {get_recommendations(tier, stats, rfm_data)}
            </div>
            """, unsafe_allow_html=True)

def get_recommendations(tier, stats, rfm_data):
    if tier == "High Value":
        return f"""
        <ul style='color: black;'>
            <li><b>VIP Treatment:</b> Offer exclusive rewards for reaching ${stats['Monetary']*1.2:,.0f} quarterly spend</li>
            <li><b>Retention Focus:</b> Personal outreach every {max(14, int(stats['Recency']/2))} days</li>
            <li><b>Premium Upsell:</b> Recommend products 30-50% higher than current average spend</li>
            <li><b>Feedback:</b> Invite to customer advisory board</li>
        </ul>
        """
    elif tier == "Medium Value":
        return f"""
        <ul style='color: black;'>
            <li><b>Upsell Strategy:</b> Bundle products to increase average order by 20-30%</li>
            <li><b>Engagement Boost:</b> Targeted emails every {max(7, int(stats['Recency']/3))} days</li>
            <li><b>Loyalty Program:</b> Offer points for reaching ${stats['Monetary']*1.5:,.0f} quarterly spend</li>
            <li><b>Feedback:</b> Survey to understand preferences</li>
        </ul>
        """
    else:
        return f"""
        <ul style='color: black;'>
            <li><b>Win-Back Campaign:</b> Special discount after {int(stats['Recency']*1.2)} days inactivity</li>
            <li><b>Low-Cost Entry:</b> Offer starter products under ${rfm_data['Monetary'].quantile(0.25):.2f}</li>
            <li><b>Reactivation:</b> "We miss you" email after {int(stats['Recency']*1.5)} days</li>
            <li><b>Survey:</b> Understand barriers to purchasing</li>
        </ul>
        """

if __name__ == "__main__":
    main()
