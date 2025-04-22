import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Advanced RFM Clustering Dashboard",
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
    .stSelectbox > div > div {
        background-color: #e6f2ff;
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


# Load data
df = load_data()

def get_value_tier(monetary_value, df):
    q25 = df['Monetary'].quantile(0.25)
    q75 = df['Monetary'].quantile(0.75)
    
    if monetary_value > q75:
        return "High Value", "#1a73e8"  # Dark blue
    elif monetary_value > q25:
        return "Medium Value", "#4285f4"  # Medium blue
    else:
        return "Low Value", "#8ab4f8"  # Light blue

def create_cluster_analysis(cluster_num, cluster_data, df, tier_color):
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
        value=cluster_data['Monetary'].mean(),
        number={'prefix': "$", 'font': {'size': 24, 'color': 'black'}},
        title={'text': f"<b>Cluster {cluster_num}</b>", 'font': {'color': 'black'}},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, df['Monetary'].max()*1.1], 'tickcolor': 'black'},
            'bar': {'color': tier_color},
            'bgcolor': 'white',
            'borderwidth': 2
        }
    ), row=1, col=1)
    
    # Cluster distribution pie chart
    cluster_counts = df['Cluster'].value_counts()
    fig.add_trace(go.Pie(
        labels=cluster_counts.index,
        values=cluster_counts.values,
        marker=dict(colors=['#1a73e8', '#4285f4', '#8ab4f8']),
        textinfo='percent+label'
    ), row=1, col=2)
    
    # RFM comparison bar chart
    fig.add_trace(go.Bar(
        x=['Recency', 'Frequency', 'Monetary'],
        y=[
            cluster_data['Recency'].mean(),
            cluster_data['Frequency'].mean(),
            cluster_data['Monetary'].mean()
        ],
        marker_color=tier_color,
        text=[
            f"{cluster_data['Recency'].mean():.1f}", 
            f"{cluster_data['Frequency'].mean():.1f}", 
            f"${cluster_data['Monetary'].mean():,.2f}"
        ],
        textposition='auto',
        textfont={'color': 'black'}
    ), row=2, col=1)
    
    # Recency vs Frequency scatter
    fig.add_trace(go.Scatter(
        x=cluster_data['Recency'],
        y=cluster_data['Frequency'],
        mode='markers',
        marker=dict(color=tier_color),
        name=f'Cluster {cluster_num}'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=700,
        title=f"<b>Cluster {cluster_num} Comprehensive Analysis</b>",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    
    return fig

def main():
    st.title("📈 Advanced RFM Clustering Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Analysis Configuration")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["BIRCH", "GMM", "KMeans"],
        key='algorithm'
    )
    
    dimension = st.sidebar.selectbox(
        "Select Dimension",
        ["RF", "FM", "RFM"],
        key='dimension'
    )
    
    # Determine cluster column based on selection
    cluster_col = f"{algorithm}_Cluster_{dimension}"
    df['Cluster'] = df[cluster_col]
    
    # Overview metrics
    st.header("📊 Cluster Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Total Clusters", df['Cluster'].nunique())
    col3.metric("Average Monetary Value", f"${df['Monetary'].mean():,.2f}")
    
    # Cluster distribution
    st.header("🔢 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    fig_pie = px.pie(
        cluster_counts,
        values=cluster_counts.values,
        names=cluster_counts.index,
        color=cluster_counts.index,
        color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'],
        hole=0.3
    )
    fig_pie.update_layout(
        title=f"{algorithm} {dimension} Cluster Distribution",
        font=dict(color='black')
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 3D Scatter plot
    st.header("🌐 Customer Segmentation")
    fig_3d = px.scatter_3d(
        df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'],
        opacity=0.7
    )
    fig_3d.update_layout(
        title=f"{algorithm} {dimension} Clustering",
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency',
            zaxis_title='Monetary ($)'
        ),
        font=dict(color='black')
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster deep dive analysis
    st.header("🔍 Cluster Deep Dive Analysis")
    
    for cluster_num in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_num]
        tier, tier_color = get_value_tier(cluster_data['Monetary'].mean(), df)
        
        with st.expander(f"Cluster {cluster_num} - {tier} ({len(cluster_data)} customers)", expanded=False):
            fig = create_cluster_analysis(cluster_num, cluster_data, df, tier_color)
            st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown(f"""
            <div class='cluster-card'>
                <h4 style='color: black;'>📋 Cluster Characteristics</h4>
                <div class='metric-box'>
                    <b>Recency:</b> {cluster_data['Recency'].mean():.1f} days ({(cluster_data['Recency'].mean()/df['Recency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Frequency:</b> {cluster_data['Frequency'].mean():.1f} purchases ({(cluster_data['Frequency'].mean()/df['Frequency'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>Monetary:</b> ${cluster_data['Monetary'].mean():,.2f} ({(cluster_data['Monetary'].mean()/df['Monetary'].mean()*100):.1f}% of average)
                </div>
                <div class='metric-box'>
                    <b>RFM Score:</b> {cluster_data['RFM_Score'].mean():.1f}
                </div>
                
                <h4 style='color: black;'>🎯 Recommended Actions</h4>
                {get_recommendations(tier, cluster_data, df)}
            </div>
            """, unsafe_allow_html=True)
    
    # Data export
    st.sidebar.markdown("---")
    st.sidebar.header("Data Export")
    if st.sidebar.button("Download Cluster Data"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download as CSV",
            data=csv,
            file_name="rfm_cluster_data.csv",
            mime="text/csv"
        )

def get_recommendations(tier, cluster_data, df):
    avg_recency = cluster_data['Recency'].mean()
    avg_frequency = cluster_data['Frequency'].mean()
    avg_monetary = cluster_data['Monetary'].mean()
    
    if tier == "High Value":
        return f"""
        <ul style='color: black;'>
            <li><b>VIP Treatment:</b> Offer exclusive rewards for reaching ${avg_monetary*1.2:,.0f} quarterly spend</li>
            <li><b>Retention Focus:</b> Personal outreach every {max(14, int(avg_recency/2))} days</li>
            <li><b>Premium Upsell:</b> Recommend products 30-50% higher than current average spend</li>
            <li><b>Feedback:</b> Invite to customer advisory board</li>
        </ul>
        """
    elif tier == "Medium Value":
        return f"""
        <ul style='color: black;'>
            <li><b>Upsell Strategy:</b> Bundle products to increase average order by 20-30%</li>
            <li><b>Engagement Boost:</b> Targeted emails every {max(7, int(avg_recency/3))} days</li>
            <li><b>Loyalty Program:</b> Offer points for reaching ${avg_monetary*1.5:,.0f} quarterly spend</li>
            <li><b>Feedback:</b> Survey to understand preferences</li>
        </ul>
        """
    else:
        return f"""
        <ul style='color: black;'>
            <li><b>Win-Back Campaign:</b> Special discount after {int(avg_recency*1.2)} days inactivity</li>
            <li><b>Low-Cost Entry:</b> Offer starter products under ${df['Monetary'].quantile(0.25):.2f}</li>
            <li><b>Reactivation:</b> "We miss you" email after {int(avg_recency*1.5)} days</li>
            <li><b>Survey:</b> Understand barriers to purchasing</li>
        </ul>
        """

if __name__ == "__main__":
    main()
