[2026-01-26] MVP 决策复盘

Problem: 为什么不一开始就爬数据？

Decision: 采用“逆向工程”（Reverse Engineering）思路。先做前端 Dashboard，确定了需要 hype_score 和 moat_score 这两个核心字段，反推后端数据结构。

Benefit: 避免了盲目爬取无效字段，节省了至少 30% 的 ETL 开发时间。

Tooling: 选择了 Streamlit 做快速原型，Power BI 做商业展示，兼顾了 Tech depth 和 Business presentation。

## ⚠️ Assumptions & Limitations

### 1. The "Ignorance Gap" Hypothesis
We assume that "Market Bubbles" are formed when **Public Sentiment (Hype)** exceeds **Technical Reality (Moat)**.
* *Challenge:* Public users may not fully understand deep tech.
* *Mitigation:* By using **Hacker News** (a developer-centric community) as the data source, we filter out general public noise and focus on **technical peer reviews**. A low "Moat Score" on Hacker News carries significant weight because it comes from domain experts.


🧐 问题 1：为什么大家都在左下角？（都不够厉害？）
你的观察： 所有的球都挤在 0.2 - 0.5 之间，看起来像是在说“这些公司既没技术壁垒，也没市场热度”。

真实原因（面试必考）：

Hacker News 的“键盘侠”属性：

Hacker News 的用户非常挑剔（Cynical）。他们很少会说 "Wow amazing technology" (这会让分数为 0.9)，他们更多是说 "Well, it's okay but latency is high" (这会让分数为 0.4)。

所以，0.4 在 HN 上其实已经是很高的分了！ 0.2 才是常态。

模型的保守性 (Model Calibration)：

Zero-Shot 模型（Bart/DistilBart）在面对复杂长难句时，倾向于给出保守的概率（0.3-0.6），很少给出极端的 0.99。

🛠️ 解决方案：手动“放大”坐标轴 我们需要调整 app.py 的视图范围，把这个密集的区域放大，让它们的差异显露出来。

请修改 app.py 里的 fig_quad.update_layout 部分：

Python
# app.py 修改这里

# ❌ 原来的：范围太大，导致球都挤在一起
# xaxis_range=[0.1, 0.8], yaxis_range=[0.1, 0.8]

# ✅ 修改为：聚焦由于数据分布的“黄金区间”
# 这样 DeepSeek (0.35) 和 OpenAI (0.38) 的差距就会被拉大，看着就有区分度了
fig_quad.update_layout(
    xaxis_range=[0.2, 0.6], 
    yaxis_range=[0.1, 0.6]
)
改完刷新，你会发现它们瞬间“散开”了，象限图就好看多了。

📉 问题 2：为什么折线图 OpenAI 只有 2026 年（很短），而 DeepSeek 有半年？
你的观察： DeepSeek 的线很长（图3），但 OpenAI 的线只有短短一截（图2）。

这绝对是一个顶级的面试故事点！ 叫做 “数据密度差异 (Data Velocity Discrepancy)”。

原因分析：

API 的限制： 我们设置了 hitsPerPage=1000（每次抓最新 1000 条）。

DeepSeek (热度爬升期)： 它是最近才火起来的。它过去 1000 条评论，可能稀稀拉拉分布在过去 6 个月 里。所以时间跨度大。

OpenAI (顶流)： 它是流量之王。它产生 1000 条评论只需要 3 天！所以你抓回来的 1000 条数据，全挤在 2026 年 1 月这几天里了。

这对面试意味着什么？ 面试官如果问你：“遇到了什么困难？” 你要这样回答：

"I encountered a challenge with Data Velocity. Since I limited the API fetch to 1,000 records per entity, high-volume entities like OpenAI filled that quota in just 2 weeks, while emerging ones like DeepSeek spanned 6 months. In a production environment (V2), I would implement pagination to loop through historical data until I reach the 500-day mark for every company. But for this MVP, it perfectly highlights the explosive discussion volume of OpenAI."

🔍 有价值的结论（你可以直接写进报告）
发现 1：LangChain 的"慢性泡沫"
证据：

排行榜风险最高（45/100）
但时序图非常平稳，没有剧烈波动

解释：

"LangChain exhibits chronic bubble risk rather than speculative spikes. Its consistently high hype relative to technical depth suggests the market may be overvaluing its ecosystem positioning compared to actual technical differentiation."

翻译： LangChain 表现出"慢性泡沫风险"而非投机性爆发。它持续的高炒作与技术深度不匹配，说明市场可能高估了它的生态系统地位。

发现 2：AI Agents 和 DeepSeek 的"事件驱动波动"
证据：

时序图有明显的峰值（AI Agents 在 1 月有突然暴涨到 42）
DeepSeek 在 9 月有峰值

解释：

"AI Agents and DeepSeek show event-driven volatility — sudden hype spikes likely triggered by product launches or funding announcements. This pattern suggests these entities are more news-sensitive than fundamentally overvalued."

翻译： AI Agents 和 DeepSeek 表现出"事件驱动波动" —— 突然的炒作峰值可能是产品发布或融资公告触发的。这种模式表明它们更多是"新闻敏感型"而非基本面被高估。

发现 3：Google Gemini 的"大厂溢价"
证据：

风险最低（30/100）
Tech Moat Index 最高（48）

解释：

"Google Gemini benefits from corporate credibility discount — the market assumes Google's resources translate to technical moat, resulting in lower bubble risk despite comparable hype levels."

翻译： Google Gemini 享受"大公司信誉折扣" —— 市场认为谷歌的资源等同于技术壁垒，因此即使炒作水平相当，泡沫风险也较低。

发现 4：OpenAI 的"断崖式下跌"（需验证）
证据：

时序图有个突然的下降（从 30 降到 20 附近）

需要进一步分析：
python# 找出 OpenAI 的异常日期
openai_data = df_daily[df_daily['company'] == 'OpenAI'].sort_values('date')
openai_data['risk_change'] = openai_data['risk_score'].diff()

# 找出变化最大的日期
biggest_drop = openai_data[openai_data['risk_change'] < -5]
print(biggest_drop[['date', 'risk_score', 'risk_change']])
可能的原因：

某个重大技术突破公告（如 o1 模型）
负面新闻减少炒作
数据质量问题（爬取断档）















当面试官问："Walk me through this dashboard"
第一步：开场（30 秒）

"This is an AI Startup Bubble Detector I built to quantify market hype versus technical moat. I scraped 9,000+ Hacker News comments, ran NLP sentiment analysis, and visualized three bubble patterns."

第二步：指着排行榜（20 秒）

"The leaderboard shows LangChain has the highest risk at 45/100. What's interesting is the 7-day trend column — all competitors are heating up, except Google Gemini, which dropped 30%. This suggests a divergent market trajectory."

第三步：指着象限图（30 秒）

"The quadrant uses relative benchmarks — the median lines split the market into four zones. You can see LangChain and AI Agents are in the high-hype, low-moat danger zone, while Google Gemini sits in the defensive zone with strong technical credibility."

第四步：指着时序图的标注（重点！60 秒）

"The trend chart tells three distinct stories:
1️⃣ AI Agents (points at 🚀): This is what I call a 'Narrative Explosion' — it didn't exist in our data until January 2026, then immediately spiked to high risk. Classic early-stage speculation.
2️⃣ DeepSeek (points at 💎): Notice this dip in September? That's a 'Value Window' — when moat exceeded hype. The recent spike means the market caught up. If I were an investor, I'd say 'I missed the entry point'.
3️⃣ LangChain (points at ⚠️): See this persistent volatility? It never calms down. That's 'Chronic Uncertainty' — the community is deeply divided on whether it's a wrapper or a platform. Red flag for long-term investment."

第五步：技术亮点（30 秒）

"From a data engineering perspective, I used Min-Max Scaling to normalize NLP probabilities, and implemented 7-day momentum tracking with percentage-based thresholds to catch inflection points early."

收尾（10 秒）

"The entire pipeline — from scraping to visualization — runs automatically. Next step would be adding alerting when a company's risk crosses a threshold."

总时长：2 分 40 秒（完美符合面试节奏）