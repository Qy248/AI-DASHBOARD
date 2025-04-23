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
    # Define recency buckets
    bins = [0, 30, 60, 90, 120, float('inf')]
    labels = ['0-30 days', '31-60 days', '61-90 days', '91-120 days', '120+ days']
    rfm_data['Recency_Bucket'] = pd.cut(rfm_data['Recency'], bins=bins, labels=labels, right=False)

    # Calculate average Frequency and Monetary for each recency bucket
    grouped = rfm_data.groupby('Recency_Bucket').agg({
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Customer ID': 'count'  # To show size of each bucket
    }).reset_index()
    grouped.rename(columns={'Customer ID': 'Customer Count'}, inplace=True)

    # Melt for line plotting
    melted = grouped.melt(id_vars=['Recency_Bucket'], value_vars=['Frequency', 'Monetary'],
                          var_name='Metric', value_name='Average Value')

    # Plot
    fig = px.line(
        melted,
        x='Recency_Bucket',
        y='Average Value',
        color='Metric',
        markers=True,
        title="Customer Frequency & Monetary Trends by Recency Bucket",
        color_discrete_sequence=['#1a73e8', '#34a853']
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=2, color='white')))
    fig.update_layout(
        xaxis_title="Recency Bucket",
        yaxis_title="Average Value",
        font=dict(color='black'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title="Metric"
    )

    return fig

def plot_clv_prediction(rfm_data):
    rfm_data['CLV'] = (rfm_data['Monetary'] * rfm_data['Frequency']) / (rfm_data['Recency'] + 1)  
    
    fig = px.box(rfm_data, x='Cluster', y='CLV', color='Cluster',
                 title='<b>Customer Lifetime Value by Cluster</b><br>Higher CLV = More Valuable Customers',
                 color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8'])
    fig.update_layout(
        yaxis_title="Estimated CLV ($)",
        showlegend=False
    )
    return fig

def plot_pca_clusters(rfm_data, model_type='KMeans', features_type='RFM', model_num=1):
    # Define features by model type
    model_features = {
        'RF': ['Recency', 'Frequency'],
        'FM': ['Frequency', 'Monetary'],
        'RFM': ['Recency', 'Frequency', 'Monetary']
    }

    # Define cluster column based on the model and features
    cluster_col = f'{model_type}Cluster{features_type}'

    # Extract features and cluster column
    features = model_features[features_type]
    cluster_col = f'{model_type}Cluster{features_type}'

    # Prepare data for PCA
    X = rfm_data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = rfm_data[cluster_col].astype(str)

    # Plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f'{model_type} PCA Cluster Distribution ({features_type} features)',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        font=dict(color='black'),
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def plot_segmentation_matrix(rfm_data):
    fig = px.scatter(
        rfm_data, 
        x='Recency', 
        y='Monetary',
        color='Cluster',
        size='Frequency',
        hover_data=['Customer ID'],
        title='<b>Recency vs. Spending</b><br>Size = Purchase Frequency',
        color_discrete_sequence=['#1a73e8', '#4285f4', '#8ab4f8']
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='black')))
    fig.update_layout(
        xaxis_title="Recency (Days Since Last Purchase)",
        yaxis_title="Total Spending ($)"
    )
    return fig


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
    
    st.header("🎯 Combined CLV + Matrix Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_clv_prediction(rfm_data), use_container_width=True)
    with col2:
        st.plotly_chart(plot_segmentation_matrix(rfm_data), use_container_width=True)
    
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
    
    # Define common styles and components
    styles = {
        "container": "margin-top: 10px;",
        "section": "padding: 12px; border-radius: 8px; margin-bottom: 10px;",
        "h4": "margin-top: 0;",
        "ul": "color: black;"
    }
    
    # Define all possible sections
    sections = {
        "high_value": {
            "color": "#1a73e8",
            "sections": [
                {
                    "title": "💎 VIP Retention Strategy",
                    "items": [
                        f"<b>Exclusive Access:</b> Invite to VIP program (Top {tier} customers spending ${monetary_value:,.2f}+)",
                        "<b>Personal Outreach:</b> Dedicated account manager with quarterly business reviews",
                        f"<b>Tiered Benefits:</b> Unlock premium features at ${monetary_value*1.2:,.0f} annual spend"
                    ],
                    "bg_color": "#e6f7ff"
                },
                {
                    "title": "🚀 Growth Opportunities",
                    "items": [
                        "<b>Premium Upsell:</b> Target with enterprise solutions 30-50% above current spend",
                        "<b>Advocacy Program:</b> Enroll in customer reference program with incentives",
                        "<b>Cross-Sell:</b> Recommend complementary premium services"
                    ],
                    "bg_color": "#f0f7ff"
                }
            ]
        },
        "medium_value": {
            "color": "#fb8c00",
            "sections": [
                {
                    "title": "📈 Value Optimization",
                    "items": [
                        "<b>Smart Bundling:</b> Create packages to increase average order by 20-30%",
                        f"<b>Loyalty Program:</b> Earn points for reaching ${monetary_value*1.5:,.0f} quarterly spend",
                        f"<b>Personalized Content:</b> Send targeted case studies every {max(7, int(recency_days/3))} days"
                    ],
                    "bg_color": "#fff3e0"
                },
                {
                    "title": "🔄 Engagement Boost",
                    "items": [
                        "<b>Usage Tips:</b> Share best practices to increase product adoption",
                        "<b>Feedback Sessions:</b> Schedule quarterly check-ins to understand needs",
                        "<b>Limited Offers:</b> Provide time-sensitive upgrades"
                    ],
                    "bg_color": "#fff8e1"
                }
            ]
        },
        "low_value": {
            "color": "#e53935",
            "sections": [
                {
                    "title": "🔙 Win-Back Strategy",
                    "items": [
                        f"<b>Special Offer:</b> {int(recency_days * 1.2)}-day reactivation discount",
                        f"<b>Re-engagement:</b> 'We've missed you' campaign after {int(recency_days * 1.5)} days",
                        f"<b>Low-Risk Entry:</b> Starter products under ${rfm_data['Monetary'].quantile(0.25):,.2f}"
                    ],
                    "bg_color": "#ffebee"
                },
                {
                    "title": "🔄 Engagement Boost",
                    "items": [
                        "<b>Usage Tips:</b> Share best practices to increase product adoption",
                        "<b>Feedback Sessions:</b> Schedule quarterly check-ins to understand needs",
                        "<b>Limited Offers:</b> Provide time-sensitive upgrades"
                    ],
                    "bg_color": "#fff8e1"
                },
                {
                    "title": "📚 Education & Support",
                    "items": [
                        "<b>Onboarding:</b> Free setup assistance and training",
                        "<b>Resource Center:</b> Curated how-to guides and tutorials",
                        "<b>Survey:</b> Identify barriers to increased usage"
                    ],
                    "bg_color": "#fce4ec"
                }
            ]
        }
    }
    
    # Select the appropriate tier
    tier_data = {
        "High Value": sections["high_value"],
        "Medium Value": sections["medium_value"],
        "Low Value": sections["low_value"]
    }[tier]
    
    # Build the HTML
    html_parts = [f"<div style='{styles['container']}'>"]
    
    for section in tier_data["sections"]:
        section_html = f"""
        <div style='background-color: {section['bg_color']}; {styles['section']}'>
            <h4 style='color: {tier_data['color']}; {styles['h4']}'>{section['title']}</h4>
            <ul style='{styles['ul']}'>
                {''.join(f'<li>{item}</li>' for item in section['items'])}
            </ul>
        </div>
        """
        html_parts.append(section_html)
    
    html_parts.append("</div>")
    
    return "".join(html_parts)


if __name__ == "__main__":
    main()
