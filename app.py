import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Startup Bubble Detector", layout="wide")

st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_process_data():
    file_path = 'data/processed_data.csv'
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None

    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hype_display'] = df['hype_score'] * 100
    df['moat_display'] = df['moat_score'] * 100
    df['risk_score'] = (df['hype_display'] * 0.6) + ((100 - df['moat_display']) * 0.4)
    
    return df

df_raw = load_and_process_data()

if df_raw is None:
    st.error("❌ Data not found. Please run nlp_engine.py first.")
    st.stop()

# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.header("🕹️ Control Panel")

# Company selector
all_companies = sorted(df_raw['company'].unique())
selected_companies = st.sidebar.multiselect(
    "Select Competitors",
    options=all_companies,
    default=all_companies 
)

# Risk score filter
st.sidebar.markdown("---")
risk_filter = st.sidebar.slider(
    "🎚️ Filter by Risk Score",
    min_value=0, 
    max_value=100, 
    value=(0, 100),
    help="Only show companies within this risk range"
)

# Apply filters
df_filtered = df_raw[
    (df_raw['company'].isin(selected_companies)) &
    (df_raw['risk_score'] >= risk_filter[0]) &
    (df_raw['risk_score'] <= risk_filter[1])
]

if len(df_filtered) == 0:
    st.warning("⚠️ No companies match your filters. Try adjusting the risk range.")
    st.stop()

# ==========================================
# Data Aggregation
# ==========================================
df_daily = (
    df_filtered
    .groupby(['company', 'date'])
    [['risk_score', 'hype_display', 'moat_display', 'sentiment_score']]
    .mean()
    .reset_index()
)

# Trend calculations
df_daily.sort_values(['company', 'date'], inplace=True)
df_daily['risk_change_7d'] = df_daily.groupby('company')['risk_score'].diff(periods=7)
df_daily['risk_change_pct'] = df_daily.groupby('company')['risk_score'].pct_change(periods=7) * 100

# ==========================================
# Forecasting Utilities
# ==========================================
def predict_next_7d(company_data):
    """
    Predict the bubble risk score 7 days ahead using linear regression.
    """
    if len(company_data) < 7:
        return np.nan
    
    recent_data = company_data.tail(30)
    
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data['risk_score'].values
    
    try:
        model = LinearRegression().fit(X, y)
        next_7d = model.predict([[len(recent_data) + 7]])[0]
        return max(0, min(100, next_7d))
    except:
        return np.nan

def get_trend_arrow(val):
    if pd.isna(val): 
        return '⚪ -'
    if val > 5:
        return f'🔴 ↑{val:.1f}%'
    if val < -5: 
        return f'🟢 ↓{val:.1f}%'
    return f'⚪ ≈{val:.1f}%'

df_latest = (
    df_daily
    .groupby('company')
    .tail(1)
    .sort_values('risk_score', ascending=False)
)

df_latest['Trend (7d)'] = df_latest['risk_change_pct'].apply(get_trend_arrow)

df_latest['Predicted (7d)'] = df_latest.apply(
    lambda row: predict_next_7d(df_daily[df_daily['company'] == row['company']]),
    axis=1
)

# ==========================================
# Main Dashboard
# ==========================================
st.title("🫧 AI Startup Bubble Detector")
st.markdown("Quantifying the gap between **Market Hype** and **Technical Moat**.")

latest_date = df_daily['date'].max()
st.caption(f"📅 Data last updated: **{latest_date}** | Tracking **{len(df_latest)}** companies")

st.subheader("🚨 Bubble Risk Leaderboard")

def highlight_risk(val):
    if val > 55:
        return 'background-color: rgba(255, 0, 0, 0.2)'
    elif val < 45:
        return 'background-color: rgba(0, 255, 0, 0.2)'
    return ''

display_cols = df_latest[
    ['company', 'risk_score', 'Trend (7d)', 'Predicted (7d)', 'hype_display', 'moat_display']
]
display_cols.columns = [
    'Company', 'Risk Score', '7-Day Trend', 'Forecast (7d)', 'Hype Prob (%)', 'Moat Prob (%)'
]

st.dataframe(
    display_cols.set_index('Company').style.map(highlight_risk, subset=['Risk Score']),
    use_container_width=True
)

csv = display_cols.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Risk Report (CSV)",
    data=csv,
    file_name=f'bubble_risk_report_{latest_date}.csv',
    mime='text/csv',
)

st.divider()

# ==========================================
# Market Quadrant
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎯 Market Quadrant")
    
    x_min, x_max = df_latest['moat_display'].min(), df_latest['moat_display'].max()
    y_min, y_max = df_latest['hype_display'].min(), df_latest['hype_display'].max()
    x_padding = max((x_max - x_min) * 0.15, 5)
    y_padding = max((y_max - y_min) * 0.15, 5)
    
    median_hype = df_latest['hype_display'].median()
    median_moat = df_latest['moat_display'].median()

    fig_quad = px.scatter(
        df_latest,
        x='moat_display',
        y='hype_display',
        color='company',
        size='risk_score',
        size_max=50,
        text='company',
        labels={
            'moat_display': 'Tech Moat Probability (%)',
            'hype_display': 'Market Hype Probability (%)'
        },
    )
    
    fig_quad.add_hline(
        y=median_hype,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text=f"Median Hype ({median_hype:.1f}%)",
        annotation_position="right"
    )
    
    fig_quad.add_vline(
        x=median_moat,
        line_dash="dash",
        line_color="green",
        opacity=0.6,
        annotation_text=f"Median Moat ({median_moat:.1f}%)",
        annotation_position="top"
    )
    
    fig_quad.update_layout(
        xaxis_range=[max(0, x_min - x_padding), min(100, x_max + x_padding)],
        yaxis_range=[max(0, y_min - y_padding), min(100, y_max + y_padding)]
    )
    
    st.plotly_chart(fig_quad, use_container_width=True)

with col2:
    st.info("""
    **Quadrant Guide:**
    - 🔴 Above median hype
    - 🟢 Above median moat
    - Dashed lines = peer group medians
    """)

# ==========================================
# Historical Trend
# ==========================================
st.subheader("📈 Historical Trend")

risk_min = df_daily['risk_score'].quantile(0.01)
risk_max = df_daily['risk_score'].quantile(0.99)
risk_padding = (risk_max - risk_min) * 0.1

fig_trend = px.line(
    df_daily,
    x='date',
    y='risk_score',
    color='company',
    markers=True,
    title="Bubble Risk Score Over Time"
)

fig_trend.update_yaxes(range=[risk_min - risk_padding, risk_max + risk_padding])
fig_trend.update_traces(connectgaps=True)

events = {
    '2024-12-10': 'Gemini 2.0 Flash',
    '2025-01-15': 'DeepSeek R1',
}

date_min = df_daily['date'].min()
date_max = df_daily['date'].max()

for date_str, event_name in events.items():
    try:
        event_date = pd.to_datetime(date_str).date()
        if date_min <= event_date <= date_max:
            fig_trend.add_shape(
                type="line",
                x0=date_str,
                x1=date_str,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
                layer="below"
            )
            fig_trend.add_annotation(
                x=date_str,
                y=1.05,
                yref="paper",
                text=event_name,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="red",
                borderwidth=1
            )
    except:
        pass

st.plotly_chart(fig_trend, use_container_width=True)

st.info("""
**📊 Pattern Recognition Guide:**
- **🚀 Narrative Explosion**: Cold start → sudden hype = early speculation
- **💎 Value Window**: Moat > hype = potential undervaluation
- **⚠️ Chronic Volatility**: Persistent debate = structural uncertainty
- **🛡️ Moat Strengthens**: Hype ↓ + Moat ↑ = defensive positioning
""")

with st.expander("🗓️ Timeline: Major AI Events (Context)"):
    st.markdown("""
    | Date | Event | Market Impact |
    |------|-------|----------------|
    | **2024-12-10** | Gemini 2.0 Flash released | Google moat strengthens |
    | **2025-01-15** | DeepSeek R1 launch | Technical validation → adoption |
    
    *Source: Hacker News comment volume & sentiment analysis*
    """)

st.markdown("---")
st.caption(
    "Built with Streamlit • Data from Hacker News via Algolia API • NLP powered by DistilBERT"
)
