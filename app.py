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

@st.cache_data
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
# 侧边栏控制
# ==========================================
st.sidebar.header("🕹️ Control Panel")

# 公司选择
all_companies = df_raw['company'].unique()
selected_companies = st.sidebar.multiselect(
    "Select Competitors",
    options=all_companies,
    default=all_companies 
)

# 🔥 功能 3: 风险过滤器
st.sidebar.markdown("---")
risk_filter = st.sidebar.slider(
    "🎚️ Filter by Risk Score",
    min_value=0, 
    max_value=100, 
    value=(0, 100),
    help="Only show companies within this risk range"
)

# 应用筛选
df_filtered = df_raw[
    (df_raw['company'].isin(selected_companies)) &
    (df_raw['risk_score'] >= risk_filter[0]) &
    (df_raw['risk_score'] <= risk_filter[1])
]

if len(df_filtered) == 0:
    st.warning("⚠️ No companies match your filters. Try adjusting the risk range.")
    st.stop()

# ==========================================
# 数据聚合
# ==========================================
df_daily = df_filtered.groupby(['company', 'date'])[['risk_score', 'hype_display', 'moat_display', 'sentiment_score']].mean().reset_index()

# 计算趋势
df_daily.sort_values(['company', 'date'], inplace=True)
df_daily['risk_change_7d'] = df_daily.groupby('company')['risk_score'].diff(periods=7)
df_daily['risk_change_pct'] = df_daily.groupby('company')['risk_score'].pct_change(periods=7) * 100

# 🔥 功能 5: 预测函数
def predict_next_7d(company_data):
    """使用线性回归预测 7 天后的风险分"""
    if len(company_data) < 7:  # 数据不足
        return np.nan
    
    # 只用最近 30 天的数据（避免长期趋势干扰）
    recent_data = company_data.tail(30)
    
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data['risk_score'].values
    
    try:
        model = LinearRegression().fit(X, y)
        next_7d = model.predict([[len(recent_data) + 7]])[0]
        return max(0, min(100, next_7d))  # 限制在 0-100
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

df_latest = df_daily.groupby('company').tail(1).sort_values('risk_score', ascending=False)
df_latest['Trend (7d)'] = df_latest['risk_change_pct'].apply(get_trend_arrow)

# 🔥 功能 5: 添加预测列
df_latest['Predicted (7d)'] = df_latest.apply(
    lambda row: predict_next_7d(df_daily[df_daily['company'] == row['company']]),
    axis=1
)

# ==========================================
# 主界面
# ==========================================
st.title("🫧 AI Startup Bubble Detector")
st.markdown("Quantifying the gap between **Market Hype** and **Technical Moat**.")

# 🔥 功能 1: 数据更新时间
latest_date = df_daily['date'].max()
st.caption(f"📅 Data last updated: **{latest_date}** | Tracking **{len(df_latest)}** companies")

st.subheader("🚨 Bubble Risk Leaderboard")

def highlight_risk(val):
    if val > 55:
        return 'background-color: rgba(255, 0, 0, 0.2)'
    elif val < 45:
        return 'background-color: rgba(0, 255, 0, 0.2)'
    return ''

# 组织排行榜列（包含预测）
display_cols = df_latest[['company', 'risk_score', 'Trend (7d)', 'Predicted (7d)', 'hype_display', 'moat_display']]
display_cols.columns = ['Company', 'Risk Score', '7-Day Trend', 'Forecast (7d)', 'Hype Prob (%)', 'Moat Prob (%)']

st.dataframe(
    display_cols.set_index('Company').style.map(highlight_risk, subset=['Risk Score']),
    use_container_width=True
)

# 🔥 功能 2: 下载报告按钮
csv = display_cols.to_csv(index=True).encode('utf-8')
st.download_button(
    label="📥 Download Risk Report (CSV)",
    data=csv,
    file_name=f'bubble_risk_report_{latest_date}.csv',
    mime='text/csv',
)

st.divider()

# ==========================================
# 象限图
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
        labels={'moat_display': 'Tech Moat Prob (%)', 'hype_display': 'Market Hype Prob (%)'},
    )
    
    fig_quad.add_hline(y=median_hype, line_dash="dash", line_color="red", opacity=0.6,
                       annotation_text=f"Median Hype ({median_hype:.1f}%)", annotation_position="right")
    fig_quad.add_vline(x=median_moat, line_dash="dash", line_color="green", opacity=0.6,
                       annotation_text=f"Median Moat ({median_moat:.1f}%)", annotation_position="top")
    
    fig_quad.update_layout(
        xaxis_range=[max(0, x_min - x_padding), min(100, x_max + x_padding)],
        yaxis_range=[max(0, y_min - y_padding), min(100, y_max + y_padding)]
    )
    st.plotly_chart(fig_quad, use_container_width=True)

with col2:
    st.info(f"""
    **Quadrant Guide:**
    - 🔴 Above median hype
    - 🟢 Above median moat
    - Lines = peer group medians
    """)

# ==========================================
# 时序图
# ==========================================
# 替换原来的时序图代码（大约第 160-180 行）
# ==========================================
# 时序图（从 "st.subheader" 开始替换到 "st.plotly_chart" 结束）
# ==========================================
st.subheader("📈 Historical Trend")

# Debug 信息（可选，检查完可以删除）
st.write("Debug: 数据日期范围")
st.write(f"最早日期: {df_daily['date'].min()}")
st.write(f"最晚日期: {df_daily['date'].max()}")

risk_min = df_daily['risk_score'].quantile(0.01)
risk_max = df_daily['risk_score'].quantile(0.99)
risk_padding = (risk_max - risk_min) * 0.1

# ==========================================
# 时序图
# ==========================================
st.subheader("📈 Historical Trend")

risk_min = df_daily['risk_score'].quantile(0.01)
risk_max = df_daily['risk_score'].quantile(0.99)
risk_padding = (risk_max - risk_min) * 0.1

# 创建折线图
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

# 🔥 修复：用 add_shape 替代 add_vline
events = {
    '2024-12-10': 'Gemini 2.0 Flash',
    '2025-01-15': 'DeepSeek R1',
}

for date_str, event_name in events.items():
    try:
        # 转换为 datetime 对象用于比较
        event_date = pd.to_datetime(date_str).date()
        
        # 检查是否在数据范围内
        if df_daily['date'].min() <= event_date <= df_daily['date'].max():
            # 🔥 方法 1：使用 add_shape 画竖线（最稳定）
            fig_trend.add_shape(
                type="line",
                x0=date_str,  # 开始位置
                x1=date_str,  # 结束位置（同一位置就是竖线）
                y0=0,         # Y 轴底部（相对坐标）
                y1=1,         # Y 轴顶部（相对坐标）
                yref="paper", # 使用相对坐标系统
                line=dict(
                    color="red",
                    width=2,
                    dash="dash"
                ),
                layer="below"  # 画在数据线下方
            )
            
            # 🔥 方法 2：单独添加标注文字
            fig_trend.add_annotation(
                x=date_str,
                y=1.05,        # 在图表顶部稍微上方
                yref="paper",  # 相对坐标
                text=event_name,
                showarrow=False,
                font=dict(size=10, color="red", family="Arial"),
                bgcolor="rgba(255, 255, 255, 0.9)",  # 白色半透明背景
                bordercolor="red",
                borderwidth=1
            )
            
            st.write(f"✅ 已添加事件标注: {event_name}")
    except Exception as e:
        st.write(f"⚠️ 无法添加事件: {event_name} ({str(e)})")

st.plotly_chart(fig_trend, use_container_width=True)

# 添加投资人视角解读
st.info("""
**📊 Pattern Recognition Guide:**
- **🚀 Narrative Explosion**: Cold start → Sudden hype = Early speculation (AI Agents pattern)
- **💎 Value Window**: Moat > Hype = Best entry point (DeepSeek Sept 2025)
- **⚠️ Chronic Volatility**: Persistent debate = Structural uncertainty (LangChain pattern)
- **🛡️ Moat Strengthens**: Hype ↓ + Moat ↑ = Defensive position (Google Gemini)
""")

# 可折叠的事件时间轴
with st.expander("🗓️ Timeline: Major AI Events (Context)"):
    st.markdown("""
    | Date | Event | Impact on Market |
    |------|-------|------------------|
    | **2024-12-10** | Gemini 2.0 Flash released | ✅ Google moat strengthens |
    | **2025-01-15** | DeepSeek R1 launch | 💎 Technical validation → Mass adoption |
    
    *Source: Hacker News comment volume & sentiment analysis*
    """)

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
st.caption("Built with Streamlit • Data from Hacker News via Algolia API • NLP powered by DistilBERT")

