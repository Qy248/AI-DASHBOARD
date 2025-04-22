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
    page_title="Advanced RFM Analytics Dashboard",
    page_icon="📈",
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
    .st-b7, .st-bb, .st-c0, .st-c1, .st-c2 {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #4285f4;
        color: white;
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

def create_comprehensive_analysis(cluster_num, stats, rfm_data):
    tier, tier_color = get_value_tier(stats['Monetary'], rfm_data)
    
    # Create subplots grid
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}, {"type": "histogram"}]
        ],
        subplot_titles=(
            "Monetary Value", 
            "Cluster Distribution", 
            "RFM Comparison",
            "Recency vs Frequency",
            "Spending Distribution",
            "Customer Value Histogram"
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # 1. Monetary value indicator
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
    
    # 2. Cluster distribution pie chart
    cluster_counts = rfm_data['Cluster'].value_counts()
    fig.add_trace(go.Pie(
        labels=cluster_counts.index,
        values=cluster_counts.values,
        marker=dict(colors=['#1a73e8', '#4285f4', '#8ab4f8']),
        textinfo='percent+label',
        hole=0.3
    ), row=1, col=2)
    
    # 3. RFM comparison bar chart
    fig.add_trace(go.Bar(
        x=['Recency', 'Frequency', 'Monetary'],
        y=[stats['Recency'], stats['Frequency'], stats['Monetary']],
        marker_color=tier_color,
        text=[f"{stats['Recency']:.1f}", f"{stats['Frequency']:.1f}", f"${stats['Monetary']:,.2f}"],
        textposition='auto',
        textfont={'color': 'black'}
    ), row=1, col=3)
    
    # 4. Recency vs Frequency scatter
    cluster_data = rfm_data[rfm_data['Cluster'] == cluster_num]
    fig.add_trace(go.Scatter(
        x=cluster_data['Recency'],
        y=cluster_data['Frequency'],
        mode='markers',
        marker=dict(color=tier_color, size=8, opacity=0.7),
        name=f'Cluster {cluster_num}'
    ), row=2, col=1)
    fig.update_xaxes(title_text="Recency (days)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    # 5. Monetary value box plot
    fig.add_trace(go.Box(
        y=cluster_data['Monetary'],
        name='Spending',
        marker_color=tier_color,
        boxmean=True
    ), row=2, col=2)
    fig.update_yaxes(title_text="Monetary Value ($)", row=2, col=2)
    
    # 6. Customer value histogram
    fig.add_trace(go.Histogram(
        x=cluster_data['Monetary'],
        nbinsx=20,
        marker_color=tier_color,
        opacity=0.7
    ), row=2, col=3)
    fig.update_xaxes(title_text="Customer Value ($)", row=2, col=3)
    fig.update_yaxes(title_text="Count", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        height=900,
        title=f"<b>Comprehensive Cluster {cluster_num} Analysis</b><br><span style='color:{tier_color}'>{tier} Customers</span>",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        margin=dict(t=100)
    )
    
    return fig, tier, tier_color

def create_trend_analysis(rfm_data):
    # Create time-based trends if date column exists
    if 'Purchase_Date' in rfm_data.columns:
        rfm_data['Purchase_Date'] = pd.to_datetime(rfm_data['Purchase_Date'])
        rfm_data['Month'] = rfm_data['Purchase_Date'].dt.to_period('M')
        
        monthly_data = rfm_data.groupby(['Month', 'Cluster']).agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Customer_ID': 'count'
        }).reset_index()
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        
        fig = px.line(
            monthly_data,
            x='Month',
            y='Monetary',
            color='Cluster',
            color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'],
            facet_col='Cluster',
            facet_col_wrap=3,
            title='Monthly Monetary Trends by Cluster',
            labels={'Monetary': 'Average Spending ($)'}
        )
        fig.update_layout(
            font=dict(color='black'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    return None

def main():
    st.title("📈 Advanced RFM Analytics Dashboard")
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
    cluster_col = f"{algorithm}_Cluster_{dimension}"
    rfm_data['Cluster'] = rfm_data[cluster_col]  # Standardize column name
    
    # Overview section
    st.header("🔍 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm_data))
    col2.metric("Total Clusters", rfm_data['Cluster'].nunique())
    col3.metric("Total Revenue", f"${rfm_data['Monetary'].sum():,.2f}")
    
    # Cluster distribution
    st.header("📊 Cluster Distribution")
    cluster_counts = rfm_data['Cluster'].value_counts().sort_index()
    
    # Cluster overview pie chart
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
        font=dict(color='black'),
        showlegend=True
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # RFM 3D scatter plot
    st.header("🌐 Customer Segmentation Overview")
    fig_3d = px.scatter_3d(
        rfm_data,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'],
        opacity=0.7,
        title='3D RFM Cluster Visualization'
    )
    fig_3d.update_layout(
        font=dict(color='black'),
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency',
            zaxis_title='Monetary ($)'
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Trend analysis (if date available)
    trend_fig = create_trend_analysis(rfm_data)
    if trend_fig:
        st.header("📅 Time-Based Trends")
        st.plotly_chart(trend_fig, use_container_width=True)
    
    # Cluster deep dive analysis
    st.header("🔬 Cluster Deep Dive Analysis")
    cluster_stats = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'Customer_ID': 'count'
    })
    cluster_stats.columns = ['Recency', 'Frequency', 'Monetary', 'Total_Revenue', 'Count']
    
    for cluster in cluster_stats.index:
        stats = cluster_stats.loc[cluster]
        fig, tier, tier_color = create_comprehensive_analysis(cluster, stats, rfm_data)
        
        with st.expander(f"📌 Cluster {cluster} - {tier} (Size: {stats['Count']} customers)", expanded=False):
            st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown(f"""
            <div class='cluster-card'>
                <h4 style='color: black;'>📋 Cluster {cluster} Characteristics</h4>
                <div class='metric-box'>
                    <b>Recency:</b> {stats['Recency']:.1f} days ({(stats['Recency']/rfm_data['Recency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Frequency:</b> {stats['Frequency']:.1f} purchases ({(stats['Frequency']/rfm_data['Frequency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Avg. Monetary:</b> ${stats['Monetary']:,.2f} ({(stats['Monetary']/rfm_data['Monetary'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Total Revenue:</b> ${stats['Total_Revenue']:,.2f} ({(stats['Total_Revenue']/rfm_data['Monetary'].sum()*100):.1f}% of total)
                </div>
                
                <h4 style='color: black;'>🎯 Strategic Recommendations</h4>
                {get_recommendations(tier, stats, rfm_data)}
            </div>
            """, unsafe_allow_html=True)
    
    # Data export
    st.sidebar.markdown("---")
    st.sidebar.header("Data Export")
    if st.sidebar.button("Download Cluster Data"):
        csv = rfm_data.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download as CSV",
            data=csv,
            file_name="rfm_cluster_data.csv",
            mime="text/csv"
        )

def get_recommendations(tier, stats, rfm_data):
    if tier == "High Value":
        return f"""
        <ul style='color: black;'>
            <li><b>VIP Program:</b> Create exclusive rewards for top {tier} customers (${stats['Monetary']:,.2f}+ spenders)</li>
            <li><b>Retention Strategy:</b> Personal outreach every {max(14, int(stats['Recency']/2))} days</li>
            <li><b>Premium Upsell:</b> Target with products 30-50% above current average spend</li>
            <li><b>Advocacy:</b> Invite to beta test new products and provide testimonials</li>
            <li><b>Loyalty:</b> Offer tiered benefits at ${stats['Monetary']*1.2:,.0f} quarterly spend level</li>
        </ul>
        """
    elif tier == "Medium Value":
        return f"""
        <ul style='color: black;'>
            <li><b>Upsell Strategy:</b> Bundle products to increase average order by 20-30%</li>
            <li><b>Engagement:</b> Targeted emails every {max(7, int(stats['Recency']/3))} days</li>
            <li><b>Loyalty:</b> Offer points for reaching ${stats['Monetary']*1.5:,.0f} quarterly spend</li>
            <li><b>Feedback:</b> Conduct preference surveys to identify upgrade opportunities</li>
            <li><b>Cross-sell:</b> Recommend complementary products based on purchase history</li>
        </ul>
        """
    else:
        return f"""
        <ul style='color: black;'>
            <li><b>Win-Back:</b> Special discount after {int(stats['Recency']*1.2)} days inactivity</li>
            <li><b>Reactivation:</b> "We miss you" campaign after {int(stats['Recency']*1.5)} days</li>
            <li><b>Entry-Level:</b> Promote starter products under ${rfm_data['Monetary'].quantile(0.25):.2f}</li>
            <li><b>Survey:</b> Understand barriers to increased purchasing</li>
            <li><b>Education:</b> Provide product usage guides and tutorials</li>
        </ul>
        """

if __name__ == "__main__":
    main()
