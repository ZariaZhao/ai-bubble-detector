import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. Configuration (Best Practice: Separation of Concerns)
# ==========================================
class Config:
    """Central configuration for easy tuning and maintenance."""
    # Data Paths
    DATA_PATH = 'data/processed_data.csv'
    
    # Risk Calculation Weights
    HYPE_WEIGHT = 0.6
    MOAT_WEIGHT = 0.4
    
    # Visual Thresholds
    HIGH_RISK_THRESHOLD = 55
    LOW_RISK_THRESHOLD = 45
    
    # Plotting Ranges (Wider lens to capture outliers)
    AXIS_RANGE = [0, 100]
    
    # Trend Analysis
    TREND_UP_THRESHOLD = 5
    TREND_DOWN_THRESHOLD = -5
    
    # Key Market Events (Contextual Markers)
    EVENTS = {
        '2024-12-10': 'Gemini 2.0 Flash',
        '2025-01-15': 'DeepSeek R1 Launch'
    }

st.set_page_config(page_title="AI Startup Bubble Detector", layout="wide")

# Custom CSS for polished look
st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
    .reportview-container .main .block-container {max-width: 1200px; padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Validation & Processing (Data Quality)
# ==========================================
def validate_dataframe(df):
    """
    Sanity checks to ensure data integrity before rendering.
    Demonstrates 'Defensive Programming'.
    """
    required_cols = {'company', 'date', 'hype_score', 'moat_score'}
    missing = required_cols - set(df.columns)
    
    if missing:
        raise ValueError(f"CRITICAL: Missing columns in dataset: {missing}")
    
    # Check for probability bounds (0-1)
    if not df['hype_score'].between(0, 1).all():
        st.warning("⚠️ Data Quality Alert: Some hype_scores are outside [0, 1] range.")
        
    # Check for date gaps (Time-series integrity)
    date_gaps = df.groupby('company')['date'].apply(
        lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).days - len(x) + 1
    )
    if date_gaps.max() > 7:
        st.toast(f"⚠️ Notice: Data gaps detected for {date_gaps[date_gaps > 7].count()} companies.", icon="ℹ️")

@st.cache_data(ttl=3600)
def load_and_process_data():
    """ETL Pipeline: Load, Validate, Transform."""
    try:
        if not os.path.exists(Config.DATA_PATH):
            st.error(f"❌ File not found: {Config.DATA_PATH}")
            return None
            
        df = pd.read_csv(Config.DATA_PATH)
        
        # 1. Validation
        validate_dataframe(df)
        
        # 2. Transformation
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['hype_display'] = df['hype_score'] * 100
        df['moat_display'] = df['moat_score'] * 100
        
        # 3. Risk Calculation
        df['risk_score'] = (
            (df['hype_display'] * Config.HYPE_WEIGHT) + 
            ((100 - df['moat_display']) * Config.MOAT_WEIGHT)
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Critical Error loading data: {str(e)}")
        return None

df_raw = load_and_process_data()

if df_raw is None:
    st.stop()

# ==========================================
# 3. Global Benchmarking (Performance Optimization)
# ==========================================
@st.cache_data(ttl=3600)
def compute_global_benchmarks(df):
    """
    Compute global medians ONCE and cache them. 
    Prevents re-calculation on every interaction.
    """
    df_daily = df.groupby(['company', 'date'])[['hype_display', 'moat_display']].mean().reset_index()
    latest = df_daily.sort_values('date').groupby('company').tail(1)
    
    return {
        'hype_median': latest['hype_display'].median(),
        'moat_median': latest['moat_display'].median()
    }

benchmarks = compute_global_benchmarks(df_raw)

# ==========================================
# 4. Advanced Analytics (ML Capability)
# ==========================================
def predict_risk_weighted(company_data):
    """
    Weighted Linear Regression with Confidence Intervals.
    Gives more weight to recent data points.
    """
    if len(company_data) < 7:
        return np.nan, np.nan
        
    recent = company_data.tail(30).copy() # Use last 30 days
    X = np.arange(len(recent)).reshape(-1, 1)
    y = recent['risk_score'].values
    
    # Exponential weights: Recent days matter more
    weights = np.exp(np.linspace(0, 1, len(recent)))
    
    try:
        model = LinearRegression().fit(X, y, sample_weight=weights)
        next_day_idx = len(recent) + 7
        pred = model.predict([[next_day_idx]])[0]
        
        # Simple Confidence Interval (Std Dev of residuals)
        residuals = y - model.predict(X)
        std_dev = np.std(residuals)
        margin_of_error = 1.96 * std_dev # 95% CI
        
        pred_clipped = max(0, min(100, pred))
        return pred_clipped, margin_of_error
    except:
        return np.nan, np.nan

def get_trend_arrow_formatted(val):
    """Formatter for trend indicators."""
    if pd.isna(val): return '⚪ N/A'
    
    if val > Config.TREND_UP_THRESHOLD:
        emoji = '🔴' # Risk increasing
        direction = '↑'
    elif val < Config.TREND_DOWN_THRESHOLD:
        emoji = '🟢' # Risk decreasing
        direction = '↓'
    else:
        emoji = '⚪'
        direction = '≈'
        
    return f"{emoji} {direction} {abs(val):.1f}%"

# ==========================================
# 5. Sidebar & Filtering
# ==========================================
st.sidebar.header("🕹️ Control Panel")

all_companies = sorted(df_raw['company'].unique())
selected_companies = st.sidebar.multiselect(
    "Select Entities",
    options=all_companies,
    default=all_companies 
)

st.sidebar.markdown("---")
risk_filter = st.sidebar.slider(
    "🎚️ Risk Filter",
    min_value=0, max_value=100, value=(0, 100),
    help="Filter entities by Risk Score. Global benchmarks remain fixed."
)

# Sidebar Stats (Data Coverage)
st.sidebar.markdown("### 📊 Data Stats")
st.sidebar.info(
    f"""
    - **Entities:** {len(all_companies)}
    - **Records:** {len(df_raw):,}
    - **Window:** {df_raw['date'].min()} to {df_raw['date'].max()}
    """
)

# Applying Filters
df_filtered = df_raw[
    (df_raw['company'].isin(selected_companies)) &
    (df_raw['risk_score'] >= risk_filter[0]) &
    (df_raw['risk_score'] <= risk_filter[1])
]

if df_filtered.empty:
    st.warning("⚠️ No data matches your filter criteria.")
    st.stop()

# ==========================================
# 6. Aggregation & Feature Engineering
# ==========================================
df_daily = (
    df_filtered
    .groupby(['company', 'date'])
    [['risk_score', 'hype_display', 'moat_display', 'sentiment_score']]
    .mean()
    .reset_index()
)
df_daily.sort_values(['company', 'date'], inplace=True)

# Calculate 7-Day Change
df_daily['risk_change_pct'] = df_daily.groupby('company')['risk_score'].pct_change(periods=7) * 100

# Get Latest Snapshot
df_latest = df_daily.groupby('company').tail(1).sort_values('risk_score', ascending=False).copy()

# Apply Advanced Analytics
df_latest['Trend_Display'] = df_latest['risk_change_pct'].apply(get_trend_arrow_formatted)

# Prediction with Unpacking
preds = df_latest.apply(
    lambda row: predict_risk_weighted(df_daily[df_daily['company'] == row['company']]), 
    axis=1
)
df_latest['Pred_Value'] = [p[0] for p in preds]
df_latest['Pred_Error'] = [p[1] for p in preds]

# Format Prediction String (e.g., "45.2 (±2.1)")
df_latest['Forecast (7d)'] = df_latest.apply(
    lambda x: f"{x['Pred_Value']:.1f} (±{x['Pred_Error']:.1f})" if pd.notnull(x['Pred_Value']) else "N/A",
    axis=1
)

# ==========================================
# 7. Dashboard Layout
# ==========================================
st.title("🫧 AI Startup Bubble Detector")
st.markdown("Quantifying the gap between **Market Hype** and **Technical Moat**.")

st.caption(f"📅 Last Updated: **{df_daily['date'].max()}**")

# --- Section A: Leaderboard ---
st.subheader("🚨 Risk Leaderboard")

def style_risk(val):
    """Conditional formatting for the dataframe."""
    if val > Config.HIGH_RISK_THRESHOLD:
        return 'background-color: rgba(255, 0, 0, 0.15); color: #8B0000'
    elif val < Config.LOW_RISK_THRESHOLD:
        return 'background-color: rgba(0, 255, 0, 0.15); color: #006400'
    return ''

display_cols = df_latest[['company', 'risk_score', 'Trend_Display', 'Forecast (7d)', 'hype_display', 'moat_display']]
display_cols.columns = ['Company', 'Risk Score', '7-Day Trend', 'Forecast (7d)', 'Hype (%)', 'Moat (%)']

st.dataframe(
    display_cols.set_index('Company').style.map(style_risk, subset=['Risk Score']).format("{:.1f}", subset=['Risk Score', 'Hype (%)', 'Moat (%)']),
    use_container_width=True
)

# CSV Download
st.download_button(
    "📥 Download Report",
    display_cols.to_csv().encode('utf-8'),
    "bubble_risk_report.csv",
    "text/csv"
)

st.divider()

# --- Section B: Quadrant Analysis ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎯 Market Quadrant (Global Benchmark)")
    
    fig_quad = px.scatter(
        df_latest,
        x='moat_display',
        y='hype_display',
        color='company',
        size='risk_score',
        size_max=50,
        text='company',
        labels={'moat_display': 'Tech Moat Probability (%)', 'hype_display': 'Market Hype Probability (%)'},
    )
    
    # Enhanced Interactivity (Tooltip)
    fig_quad.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "Moat: %{x:.1f}%<br>" +
                      "Hype: %{y:.1f}%<br>" +
                      "Risk: %{marker.size:.1f}<br>" +
                      "<extra></extra>"
    )
    
    # Fixed Benchmarks
    fig_quad.add_hline(y=benchmarks['hype_median'], line_dash="dash", line_color="red", opacity=0.5, annotation_text=f"Global Hype ({benchmarks['hype_median']:.1f}%)")
    fig_quad.add_vline(x=benchmarks['moat_median'], line_dash="dash", line_color="green", opacity=0.5, annotation_text=f"Global Moat ({benchmarks['moat_median']:.1f}%)")
    
    # Polished Layout
    fig_quad.update_layout(
        xaxis_range=Config.AXIS_RANGE,
        yaxis_range=Config.AXIS_RANGE,
        template="plotly_white", # Clean white background
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_quad, use_container_width=True)

with col2:
    # Methodology Expander
    with st.expander("🔍 Methodology", expanded=True):
        st.markdown(f"""
        **Risk Score Formula:**
        $$Risk = {Config.HYPE_WEIGHT}\\times Hype + {Config.MOAT_WEIGHT}\\times(100-Moat)$$
        
        **Definitions:**
        - **Hype:** Marketing buzz & speculation.
        - **Moat:** Technical depth & reproducibility.
        - **Benchmarks:** Median of all {len(all_companies)} entities.
        """)
        st.info("Use the sidebar filter to focus on specific risk profiles.")

# --- Section C: Historical Trend ---
st.subheader("📈 Historical Trend")

fig_trend = px.line(
    df_daily,
    x='date',
    y='risk_score',
    color='company',
    markers=True,
    title="Risk Score Evolution"
)

# Add Event Markers
for date_str, event_name in Config.EVENTS.items():
    try:
        fig_trend.add_vline(
            x=date_str, 
            line_dash="dot", 
            line_color="gray", 
            annotation_text=event_name,
            annotation_position="top left"
        )
    except: pass

fig_trend.update_layout(
    yaxis_range=[20, 80], # Focused view
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig_trend, use_container_width=True)