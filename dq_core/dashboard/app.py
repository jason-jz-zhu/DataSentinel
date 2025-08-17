"""Streamlit dashboard for DataSentinel."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


# Page config
st.set_page_config(
    page_title="DataSentinel Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.failed-card {
    border-left-color: #d62728;
}
.passed-card {
    border-left-color: #2ca02c;
}
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    st.title("🛡️ DataSentinel Dashboard")
    st.markdown("Enterprise Data Quality Monitoring")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "Datasets", "Data Quality Scores", "Trends", "Alerts"]
        )
        
        st.header("Filters")
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )
    
    if page == "Overview":
        show_overview()
    elif page == "Datasets":
        show_datasets()
    elif page == "Data Quality Scores":
        show_scores()
    elif page == "Trends":
        show_trends()
    elif page == "Alerts":
        show_alerts()


def show_overview():
    """Show overview dashboard."""
    st.header("Data Quality Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall DQ Score",
            value="85%",
            delta="2.3%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Datasets Monitored",
            value="12",
            delta="1",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Checks Passing",
            value="156/180",
            delta="-3",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Critical Issues",
            value="2",
            delta="-1",
            delta_color="normal"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("DQ Score by Dimension")
        
        dimensions = ["Accuracy", "Reliability", "Stewardship", "Usability"]
        scores = [90, 80, 75, 85]
        
        fig = px.bar(
            x=dimensions,
            y=scores,
            title="Data Quality Dimensions",
            color=scores,
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Check Status Distribution")
        
        status_data = pd.DataFrame({
            "Status": ["Passed", "Failed", "Warning"],
            "Count": [156, 24, 8],
            "Color": ["#2ca02c", "#d62728", "#ff7f0e"]
        })
        
        fig = px.pie(
            status_data,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map={
                "Passed": "#2ca02c",
                "Failed": "#d62728",
                "Warning": "#ff7f0e"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Scan Activity")
    activity_data = pd.DataFrame({
        "Dataset": ["customers", "transactions", "products", "orders"],
        "Last Scan": ["2 hours ago", "1 hour ago", "30 minutes ago", "15 minutes ago"],
        "Score": [85, 92, 78, 88],
        "Status": ["✅ Passed", "✅ Passed", "⚠️ Warning", "✅ Passed"]
    })
    st.dataframe(activity_data, use_container_width=True)


def show_datasets():
    """Show datasets page."""
    st.header("Dataset Management")
    
    # Dataset list
    datasets_data = pd.DataFrame({
        "Dataset": ["customers", "transactions", "products", "orders", "inventory"],
        "Storage Type": ["Snowflake", "S3", "Spark", "Snowflake", "S3"],
        "Owner": ["data-team", "analytics", "product", "sales", "operations"],
        "Last Scan": ["2023-11-15 10:30", "2023-11-15 09:15", "2023-11-15 08:45", "2023-11-15 11:00", "2023-11-15 07:30"],
        "DQ Score": [85, 92, 78, 88, 82],
        "Status": ["🟢 Healthy", "🟢 Healthy", "🟡 Warning", "🟢 Healthy", "🟢 Healthy"]
    })
    
    st.dataframe(datasets_data, use_container_width=True)
    
    # Dataset details
    selected_dataset = st.selectbox("Select Dataset for Details", datasets_data["Dataset"])
    
    if selected_dataset:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_dataset} - Profile")
            profile_data = pd.DataFrame({
                "Column": ["id", "name", "email", "created_at", "status"],
                "Type": ["INTEGER", "VARCHAR", "VARCHAR", "TIMESTAMP", "VARCHAR"],
                "Null %": [0.0, 0.5, 2.1, 0.0, 0.8],
                "Distinct": [1000, 998, 976, 1000, 3]
            })
            st.dataframe(profile_data)
        
        with col2:
            st.subheader(f"{selected_dataset} - Recent Checks")
            checks_data = pd.DataFrame({
                "Check": ["Not Null ID", "Unique Email", "Valid Status", "Recent Data"],
                "Status": ["✅ Pass", "⚠️ Warning", "✅ Pass", "✅ Pass"],
                "Pass Rate": ["100%", "97.9%", "100%", "100%"]
            })
            st.dataframe(checks_data)


def show_scores():
    """Show DQ scores page."""
    st.header("Data Quality Scores")
    
    # Score summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Score", "84.2%", "1.5%")
    with col2:
        st.metric("Best Performing", "transactions (92%)")
    with col3:
        st.metric("Needs Attention", "products (78%)")
    
    # Detailed scores
    scores_data = pd.DataFrame({
        "Dataset": ["customers", "transactions", "products", "orders", "inventory"],
        "Overall": [85, 92, 78, 88, 82],
        "Accuracy": [90, 95, 75, 85, 80],
        "Reliability": [80, 90, 80, 90, 85],
        "Stewardship": [75, 85, 70, 80, 75],
        "Usability": [85, 95, 85, 95, 90]
    })
    
    # Heatmap
    st.subheader("Score Heatmap")
    fig = px.imshow(
        scores_data.set_index("Dataset"),
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Scores")
    st.dataframe(scores_data, use_container_width=True)


def show_trends():
    """Show trends page."""
    st.header("Data Quality Trends")
    
    # Generate sample trend data
    dates = pd.date_range(start="2023-10-01", end="2023-11-15", freq="D")
    
    trends_data = pd.DataFrame({
        "Date": dates,
        "customers": np.random.normal(85, 5, len(dates)),
        "transactions": np.random.normal(90, 3, len(dates)),
        "products": np.random.normal(80, 7, len(dates)),
        "orders": np.random.normal(88, 4, len(dates))
    })
    
    # Ensure scores stay within 0-100
    for col in ["customers", "transactions", "products", "orders"]:
        trends_data[col] = np.clip(trends_data[col], 0, 100)
    
    st.subheader("DQ Score Trends (Last 45 Days)")
    
    fig = go.Figure()
    for dataset in ["customers", "transactions", "products", "orders"]:
        fig.add_trace(go.Scatter(
            x=trends_data["Date"],
            y=trends_data[dataset],
            mode="lines+markers",
            name=dataset
        ))
    
    fig.update_layout(
        title="Data Quality Score Trends",
        xaxis_title="Date",
        yaxis_title="DQ Score (%)",
        yaxis=dict(range=[60, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.subheader("Trend Analysis")
    analysis_data = pd.DataFrame({
        "Dataset": ["customers", "transactions", "products", "orders"],
        "Current Score": [85, 92, 78, 88],
        "30-day Avg": [83, 90, 80, 87],
        "Trend": ["📈 +2.4%", "📈 +2.2%", "📉 -2.5%", "📈 +1.1%"],
        "Volatility": ["Low", "Low", "High", "Medium"]
    })
    st.dataframe(analysis_data, use_container_width=True)


def show_alerts():
    """Show alerts page."""
    st.header("Data Quality Alerts")
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Open Alerts", "5", "-2")
    with col2:
        st.metric("Critical", "2", "0")
    with col3:
        st.metric("Resolved Today", "8", "+3")
    
    # Active alerts
    st.subheader("Active Alerts")
    alerts_data = pd.DataFrame({
        "Severity": ["🔴 Critical", "🟡 Warning", "🔴 Critical", "🟡 Warning", "🟠 Medium"],
        "Dataset": ["products", "customers", "inventory", "orders", "transactions"],
        "Check": ["Null values spike", "Email format", "Stock count anomaly", "Late arrival", "Duplicate IDs"],
        "Triggered": ["2h ago", "4h ago", "1h ago", "6h ago", "3h ago"],
        "Owner": ["product", "data-team", "operations", "sales", "analytics"]
    })
    
    st.dataframe(alerts_data, use_container_width=True)
    
    # Alert timeline
    st.subheader("Alert Timeline (Last 7 Days)")
    
    timeline_dates = pd.date_range(start="2023-11-08", end="2023-11-15", freq="D")
    timeline_data = pd.DataFrame({
        "Date": timeline_dates,
        "Critical": np.random.poisson(0.5, len(timeline_dates)),
        "Warning": np.random.poisson(1.5, len(timeline_dates)),
        "Medium": np.random.poisson(2, len(timeline_dates))
    })
    
    fig = px.bar(
        timeline_data,
        x="Date",
        y=["Critical", "Warning", "Medium"],
        title="Daily Alert Count",
        color_discrete_map={
            "Critical": "#d62728",
            "Warning": "#ff7f0e", 
            "Medium": "#2ca02c"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()