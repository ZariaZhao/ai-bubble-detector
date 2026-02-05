import pandas as pd
from transformers import pipeline
from textblob import TextBlob
import time

# ==========================================
# 配置区
# ==========================================
INPUT_FILE = 'data/raw_hn_data.csv'
OUTPUT_FILE = 'data/processed_data.csv' # 也就是我们要喂给 Dashboard 的最终数据
SAMPLE_SIZE = None  # ⚠️ 测试阶段：每家公司只跑50条，防止电脑跑太久。正式跑可以设为 None

# 定义我们的核心维度 (Labels)
CANDIDATE_LABELS = [
    "Technical Deep Dive",      # 代表 Moat (壁垒)
    "Marketing Hype",           # 代表 Hype (炒作)
    "Business Model Analysis",  # 代表 Monetization (变现)
    "Real World Application"    # 代表 Adoption (落地)
]

def analyze_sentiment(text):
    """
    计算情感极性 (-1: 负面, 1: 正面)
    """
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

def run_nlp_pipeline():
    print("🚀 Loading NLP Model (Zero-Shot Classification)...")
    print("   (This might take a while for the first time download...)")
    
    # 加载 HuggingFace 模型 (CPU 模式)
    # 这里的 model='facebook/bart-large-mnli' 是业界标准的 Zero-Shot 模型
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # 读取原始数据
    df = pd.read_csv(INPUT_FILE)
    print(f"📊 Loaded {len(df)} raw comments.")
    
    # ⚠️ 采样：为了快速验证，我们先只取前 N 条跑通流程
    if SAMPLE_SIZE:
        df = df.groupby('company').head(SAMPLE_SIZE).reset_index(drop=True)
        print(f"✂️ Subsampled to {SAMPLE_SIZE} comments per company for speed.")

    results = []
    
    print("🧠 Start Scoring Comments (Grab a coffee, this takes time)...")
    start_time = time.time()
    
    for index, row in df.iterrows():
        text = str(row['comment_text'])[:512] # 截断一下，防止太长报错
        
        # 1. 情感分析 (Sentiment)
        sentiment = analyze_sentiment(text)
        
        # 2. 维度分类 (Zero-Shot)
        # 这一步是让 AI 判断这句话属于哪个类别
        classification = classifier(text, CANDIDATE_LABELS, multi_label=False)
        
        # 提取分数
        scores = dict(zip(classification['labels'], classification['scores']))
        
        # 映射到我们的 Bubble Index 字段
        hype_prob = scores.get("Marketing Hype", 0)
        moat_prob = scores.get("Technical Deep Dive", 0)
        
        # 3. 计算 Bubble Index (核心公式)
        # 逻辑：如果是正面情感且被归类为 Hype -> 增加泡沫分
        #      如果是正面情感且被归类为 Moat -> 减少泡沫分 (增加壁垒)
        #      这里做一个简化版公式：
        bubble_index = (hype_prob * 0.7) + (0.3 * (1 - moat_prob)) 
        
        results.append({
            'date': row['date'],
            'company': row['company'],
            'comment_text': row['comment_text'], # 保留原文方便展示 Top Comments
            'sentiment_score': sentiment,
            'hype_score': hype_prob,
            'moat_score': moat_prob,
            'bubble_index': bubble_index,
            
            # 保留详细概率给 Power BI 用
            'marketing_hype_prob': hype_prob,
            'technical_moat_prob': moat_prob,
            'monetization_prob': scores.get("Business Model Analysis", 0),
            'real_world_usage_prob': scores.get("Real World Application", 0)
        })
        
        # 打印进度条
        if index % 10 == 0:
            print(f"   Processed {index}/{len(df)} comments...")

    # 保存结果
    final_df = pd.read_json(pd.DataFrame(results).to_json()) # 简单的格式清洗
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    end_time = time.time()
    print(f"\n✅ NLP Pipeline Completed in {round(end_time - start_time, 2)} seconds.")
    print(f"💾 Processed data saved to {OUTPUT_FILE}. Ready for Dashboard!")

if __name__ == "__main__":
    run_nlp_pipeline()