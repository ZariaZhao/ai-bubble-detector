import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# ==========================================
# 配置区
# ==========================================
# 我们要关注的公司/关键词
KEYWORDS = [
    'OpenAI',        # 基准 (Benchmark)
    'Google Gemini', # 大厂竞品 (Big Tech)
    'DeepSeek',      # 🔥 当前最热 (Trend / High Moat?)
    'Perplexity',    # 应用层代表 (App Layer)
    'LangChain',     # 中间件 (Middleware / "Wrapper" Debate)
    'AI Agents'      # 热门话题 (Future Narrative)
]
# 爬取过去多少天的数据
DAYS_BACK = 500
# 每次请求的间隔（秒），防止被封 IP
SLEEP_TIME = 1

def fetch_hn_data(keyword, days_back):
    """
    使用 Algolia API 搜索 Hacker News 上的相关评论
    """
    print(f"🔍 Searching for: {keyword}...")
    
    # 计算时间戳
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    numeric_filter = f"created_at_i>{int(start_date.timestamp())}"
    
    # Hacker News Algolia API URL
    url = "http://hn.algolia.com/api/v1/search_by_date"
    
    params = {
        'query': keyword,
        'tags': 'comment',       # 只抓评论，不抓新闻标题
        'numericFilters': numeric_filter,
        'hitsPerPage': 1000       # 改大！让数据更密集，时间跨度更长
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # 提取有用字段
        comments = []
        for hit in data['hits']:
            comments.append({
                'date': hit['created_at'],
                'company': keyword,
                'comment_text': hit['comment_text'], # 还没清洗的原始文本
                'author': hit['author'],
                'points': hit.get('points', 0), # 爬取点赞数，如果没有就填0
                'objectID': hit['objectID']
                
            })
        
        print(f"   ✅ Found {len(comments)} comments.")
        return comments
    
    except Exception as e:
        print(f"   ❌ Error fetching {keyword}: {e}")
        return []

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    all_data = []
    
    print("🚀 Starting Data Pipeline...")
    
    for company in KEYWORDS:
        company_data = fetch_hn_data(company, DAYS_BACK)
        all_data.extend(company_data)
        time.sleep(SLEEP_TIME) # 礼貌爬虫
    
    # 转为 DataFrame
    df = pd.DataFrame(all_data)
    
    # 简单清洗：去除空文本
    df = df[df['comment_text'].notna()]
    
    # 转换时间格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 预览
    print("\n📊 Data Summary:")
    print(df.groupby('company').size())
    
    # 保存为“原始数据” (Raw Data)
    # 注意：这里我们存为 raw_hn_data.csv，不覆盖之前的 mock_data.csv
    # 因为这个文件里还没有 score，直接喂给 Dashboard 会报错
    filename = 'data/raw_hn_data.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 Raw data saved to {filename}. Next step: NLP Scoring.")