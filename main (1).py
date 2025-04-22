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
            'Customer ID': 'count'  # Changed to match your column name
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
    
    trend_fig = create_trend_analysis(rfm_data)
    if trend_fig:
        st.header("📅 Time-Based Trends")
        st.plotly_chart(trend_fig, use_container_width=True)
    
    st.header("🔬 Cluster Deep Dive Analysis")
    
    # Assuming rfm_data is already loaded
    cluster_stats = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'Customer ID': 'count'  # Changed to match your column name
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
            </div>
            """, unsafe_allow_html=True)

            # Recommended engagement strategy
            st.markdown(f"""
            <h3 style='color: {tier_color}; border-bottom: 2px solid {tier_color}; padding-bottom: 8px;'>
                Recommended Engagement Strategy
            </h3>
            {get_enhanced_recommendations(tier, stats, rfm_data)}
            """, unsafe_allow_html=True)
    
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


# Helper functions
def get_comparison_arrow(value, average):
    if value > average:
        return "↑ Above average"
    elif value < average:
        return "↓ Below average"
    else:
        return "→ At average"

def get_enhanced_recommendations(tier, stats, rfm_data):
    recency_days = int(stats['Recency'])
    monetary_value = stats['Monetary']
    
    if tier == "High Value":
        return f"""
        <div style='margin-top: 10px;'>
            <div style='background-color: #e6f7ff; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                <h4 style='color: #1a73e8; margin-top: 0;'>💎 VIP Retention Strategy</h4>
                <ul style='color: black;'>
                    <li><b>Exclusive Access:</b> Invite to VIP program (Top {tier} customers spending ${monetary_value:,.2f}+)</li>
                    <li><b>Personal Outreach:</b> Dedicated account manager with quarterly business reviews</li>
                    <li><b>Tiered Benefits:</b> Unlock premium features at ${monetary_value*1.2:,.0f} annual spend</li>
                </ul>
            </div>
            
            <div style='background-color: #f0f7ff; padding: 12px; border-radius: 8px;'>
                <h4 style='color: #1a73e8; margin-top: 0;'>🚀 Growth Opportunities</h4>
                <ul style='color: black;'>
                    <li><b>Premium Upsell:</b> Target with enterprise solutions 30-50% above current spend</li>
                    <li><b>Advocacy Program:</b> Enroll in customer reference program with incentives</li>
                    <li><b>Cross-Sell:</b> Recommend complementary premium services</li>
                </ul>
            </div>
        </div>
        """
    elif tier == "Medium Value":
        return f"""
        <div style='margin-top: 10px;'>
            <div style='background-color: #fff3e0; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                <h4 style='color: #fb8c00; margin-top: 0;'>📈 Value Optimization</h4>
                <ul style='color: black;'>
                    <li><b>Smart Bundling:</b> Create packages to increase average order by 20-30%</li>
                    <li><b>Loyalty Program:</b> Earn points for reaching ${monetary_value*1.5:,.0f} quarterly spend</li>
                    <li><b>Personalized Content:</b> Send targeted case studies every {max(7, int(recency_days/3))} days</li>
                </ul>
            </div>
            
            <div style='background-color: #fff8e1; padding: 12px; border-radius: 8px;'>
                <h4 style='color: #fb8c00; margin-top: 0;'>🔄 Engagement Boost</h4>
                <ul style='color: black;'>
                    <li><b>Usage Tips:</b> Share best practices to increase product adoption</li>
                    <li><b>Feedback Sessions:</b> Schedule quarterly check-ins to understand needs</li>
                    <li><b>Limited Offers:</b> Provide time-sensitive upgrades</li>
                </ul>
            </div>
        </div>
        """
    else:
        winback_days = int(recency_days * 1.2)
        reactivation_days = int(recency_days * 1.5)
        entry_point = rfm_data['Monetary'].quantile(0.25)
        
        return f"""
        <div style='margin-top: 10px;'>
            <div style='background-color: #ffebee; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                <h4 style='color: #e53935; margin-top: 0;'>🔙 Win-Back Strategy</h4>
                <ul style='color: black;'>
                    <li><b>Special Offer:</b> {winback_days}-day reactivation discount</li>
                    <li><b>Re-engagement:</b> "We've missed you" campaign after {reactivation_days} days</li>
                    <li><b>Low-Risk Entry:</b> Starter products under ${entry_point:,.2f}</li>
                </ul>
            </div>
            
            <div style='background-color: #fff8e1; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                <h4 style='color: #fb8c00; margin-top: 0;'>🔄 Engagement Boost</h4>
                <ul style='color: black;'>
                    <li><b>Usage Tips:</b> Share best practices to increase product adoption</li>
                    <li><b>Feedback Sessions:</b> Schedule quarterly check-ins to understand needs</li>
                    <li><b>Limited Offers:</b> Provide time-sensitive upgrades</li>
                </ul>
            </div>
            
            <div style='background-color: #fce4ec; padding: 12px; border-radius: 8px;'>
                <h4 style='color: #e53935; margin-top: 0;'>📚 Education & Support</h4>
                <ul style='color: black;'>
                    <li><b>Onboarding:</b> Free setup assistance and training</li>
                    <li><b>Resource Center:</b> Curated how-to guides and tutorials</li>
                    <li><b>Survey:</b> Identify barriers to increased usage</li>
                </ul>
            </div>
        </div>
        """


if __name__ == "__main__":
    main()
