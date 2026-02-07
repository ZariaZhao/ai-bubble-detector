import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. Config & Data Loading
# ==========================================
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
    # Convert raw probability to percentage (0-100)
    df['hype_display'] = df['hype_score'] * 100
    df['moat_display'] = df['moat_score'] * 100
    
    # Risk Score Formula
    df['risk_score'] = (df['hype_display'] * 0.6) + ((100 - df['moat_display']) * 0.4)
    
    return df

df_raw = load_and_process_data()

if df_raw is None:
    st.error("❌ Data not found. Please run nlp_engine.py first.")
    st.stop()

# ==========================================
# 2. Global Benchmark Calculation (The Fix)
# ==========================================
# 🔥 CRITICAL FIX: Calculate "Global" averages before applying any filters.
# This ensures the benchmark lines (dashed) remain static regardless of user filtering.
df_global_daily = df_raw.groupby(['company', 'date'])[['hype_display', 'moat_display']].mean().reset_index()
df_global_latest = df_global_daily.sort_values('date').groupby('company').tail(1)

GLOBAL_HYPE_MEDIAN = df_global_latest['hype_display'].median()
GLOBAL_MOAT_MEDIAN = df_global_latest['moat_display'].median()

# ==========================================
# 3. Sidebar Controls (Filtering)
# ==========================================
st.sidebar.header("🕹️ Control Panel")

all_companies = sorted(df_raw['company'].unique())
selected_companies = st.sidebar.multiselect(
    "Select Competitors",
    options=all_companies,
    default=all_companies 
)

st.sidebar.markdown("---")
risk_filter = st.sidebar.slider(
    "🎚️ Filter by Risk Score",
    min_value=0, 
    max_value=100, 
    value=(0, 100),
    help="Filter entities, but keep the global benchmark lines fixed."
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
# 4. Data Aggregation (Filtered)
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
df_daily['risk_change_pct'] = df_daily.groupby('company')['risk_score'].pct_change(periods=7) * 100

# Prediction Logic
def predict_next_7d(company_data):
    if len(company_data) < 7: return np.nan
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
    if pd.isna(val): return '⚪ -'
    if val > 5: return f'🔴 ↑{val:.1f}%'
    if val < -5: return f'🟢 ↓{val:.1f}%'
    return f'⚪ ≈{val:.1f}%'

df_latest = (
    df_daily
    .groupby('company')
    .tail(1)
    .sort_values('risk_score', ascending=False)
)

df_latest['Trend (7d)'] = df_latest['risk_change_pct'].apply(get_trend_arrow)
df_latest['Predicted (7d)'] = df_latest.apply(
    lambda row: predict_next_7d(df_daily[df_daily['company'] == row['company']]), axis=1
)

# ==========================================
# 5. Dashboard UI
# ==========================================
st.title("🫧 AI Startup Bubble Detector")
st.markdown("Quantifying the gap between **Market Hype** and **Technical Moat**.")

latest_date = df_daily['date'].max()
st.caption(f"📅 Data updated: **{latest_date}** | Global Benchmark based on {len(all_companies)} entities")

# --- Row 1: Leaderboard ---
st.subheader("🚨 Bubble Risk Leaderboard")

def highlight_risk(val):
    if val > 55: return 'background-color: rgba(255, 0, 0, 0.2)'
    elif val < 45: return 'background-color: rgba(0, 255, 0, 0.2)'
    return ''

display_cols = df_latest[['company', 'risk_score', 'Trend (7d)', 'Predicted (7d)', 'hype_display', 'moat_display']]
display_cols.columns = ['Company', 'Risk Score', '7-Day Trend', 'Forecast (7d)', 'Hype Prob (%)', 'Moat Prob (%)']

st.dataframe(
    display_cols.set_index('Company').style.map(highlight_risk, subset=['Risk Score']),
    use_container_width=True
)

# CSV Download
csv = display_cols.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Risk Report (CSV)",
    data=csv,
    file_name=f'bubble_risk_report_{latest_date}.csv',
    mime='text/csv',
)

st.divider()

# --- Row 2: Quadrant Chart (Fixed) ---
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
    
    # 🔥 CRITICAL FIX: Hardcode the benchmark lines using Global Medians
    fig_quad.add_hline(
        y=GLOBAL_HYPE_MEDIAN, line_dash="dash", line_color="red", opacity=0.5,
        annotation_text=f"Global Median Hype ({GLOBAL_HYPE_MEDIAN:.1f}%)", annotation_position="top left"
    )
    
    fig_quad.add_vline(
        x=GLOBAL_MOAT_MEDIAN, line_dash="dash", line_color="green", opacity=0.5,
        annotation_text=f"Global Median Moat ({GLOBAL_MOAT_MEDIAN:.1f}%)", annotation_position="top left"
    )
    
    # 🔥 CRITICAL FIX: Lock the axis ranges to [10, 70]
    # This ensures entities (like DeepSeek) stay in the exact same position, preventing visual jumping when others are filtered out.
    fig_quad.update_layout(
        xaxis_range=[10, 70], 
        yaxis_range=[10, 70],
        showlegend=True
    )
    
    st.plotly_chart(fig_quad, use_container_width=True)

with col2:
    st.info(f"""
    **Global Benchmarks:**
    
    - **Hype Median:** {GLOBAL_HYPE_MEDIAN:.1f}%
    - **Moat Median:** {GLOBAL_MOAT_MEDIAN:.1f}%
    
    *Benchmarks are calculated from the full dataset and remain fixed when filtering.*
    """)

# --- Row 3: Historical Trend ---
st.subheader("📈 Historical Trend")

# Unify Y-axis range for consistent comparison
risk_min, risk_max = 20, 80 

fig_trend = px.line(
    df_daily,
    x='date',
    y='risk_score',
    color='company',
    markers=True,
    title="Bubble Risk Score Over Time"
)

fig_trend.update_yaxes(range=[risk_min, risk_max])
fig_trend.update_traces(connectgaps=True)

# Add Events
events = {'2024-12-10': 'Gemini 2.0', '2025-01-15': 'DeepSeek R1'}
for date_str, event_name in events.items():
    try:
        fig_trend.add_vline(x=date_str, line_dash="dot", annotation_text=event_name, line_color="gray")
    except: pass

st.plotly_chart(fig_trend, use_container_width=True)