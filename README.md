## 📅 Project Roadmap & Status

### Phase 1: MVP & Architecture Verification (✅ Completed)
- [x] Designed the "Bubble Index" quantification framework (Hype vs. Utility).
- [x] Built the interactive dashboard using **Streamlit** to validate visualization logic.
- [x] Established the data schema and integrated with **Power BI** for business reporting.
- [x] **Outcome:** Validated the mathematical feasibility of the quadrant model using simulated Gaussian distribution data.

### Phase 2: Data Pipeline Engineering (🚧 In Progress)
- [ ] Implement ETL pipeline to scrape unstructured discussions from **Hacker News (Algolia API)**.
- [ ] Develop NLP preprocessing module (Tokenization, Noise Removal).
- [ ] Store raw and processed data into **SQLite/CSV** for auditability.

### Phase 3: NLP Core & Inference (Next Step)
- [ ] Deploy Zero-Shot Classification model (BART-Large-MNLI) to score "Moat" vs. "Hype".

# AI Bubble Detector
*NLP-Driven Market Hype & Narrative Analysis*

---

## 📌 Project Overview

Early-stage AI companies and emerging AI concepts are difficult to evaluate using traditional financial metrics.  
Meanwhile, online discussions are rich but unstructured.

This project builds an end-to-end NLP-driven analytics pipeline that transforms public discussions into structured, interpretable indicators such as **Hype**, **Technical Moat**, and a composite **Bubble Index**.

---

## 🎯 Problem Statement

- Financial data is limited or unreliable for early AI innovations
- Public sentiment exists but lacks structure
- Market narratives often blur hype and substance

---

## 💡 Solution

An analytics framework that:
- Collects Hacker News discussions via API
- Uses zero-shot NLP to classify discussion intent
- Quantifies hype-heavy versus substance-driven narratives
- Visualizes trends through dashboards and BI tools

---

## 🧠 Methodology & Pipeline

1. **Data Acquisition**
   - Source: Hacker News (Algolia API)
   - Entity types:
     - Companies: OpenAI, Google Gemini, Perplexity, LangChain
     - Topics: DeepSeek, AI Agents
   - Per-entity comment cap: 100 (for rapid iteration)

2. **Data Cleaning**
   - Remove empty or deleted comments
   - Basic text normalization
   - Deduplication by comment ID

3. **NLP Feature Engineering**
   - Zero-shot classification (`facebook/bart-large-mnli`)
   - Candidate labels:
     - Technical Moat
     - Marketing Hype
     - Monetization Strategy
     - Real World Usage
   - Sentiment analysis (VADER / TextBlob)

4. **Metric Construction**
   - **Hype Score:** Probability of marketing-driven or speculative language
   - **Moat Score:** Probability of technical or defensible discussion
   - **Bubble Index:** Composite signal capturing hype–substance imbalance

5. **Visualization**
   - Streamlit interactive dashboard
   - PowerBI-ready CSV export

---

## 📊 Key Metrics Explained

- **Hype Score**  
  Measures how strongly discussions emphasize speculative optimism or marketing narratives.

- **Moat Score**  
  Measures the presence of technical depth, constraints, or defensible capabilities.

- **Bubble Index**  
  Indicates imbalance where hype dominates substance.

> This system analyzes *how people talk*, not whether they are correct.

---

## 📈 Dashboard Preview

*(Insert screenshots here)*

- Bubble Index over time
- Hype vs Moat quadrant
- Comment-level explainability

---

## ✅ Evaluation Strategy

Due to the absence of labeled ground truth, evaluation focuses on:

- **Face Validity**  
  Known contrasts between established platforms and hype-driven products

- **Temporal Consistency**  
  Scores should not fluctuate drastically day-to-day

- **Human-in-the-loop Validation**  
  Manual review of high-hype samples

---

## ⚠️ Limitations

- Reflects public perception rather than internal technical reality
- Discussion volume varies by entity
- Topic-level analysis abstracts away company-specific details

---

## 🛠️ Tech Stack

- Python
- Hugging Face Transformers
- Streamlit
- PowerBI
- Pandas, Requests

---

## 📂 Project Structure

```text
ai-bubble-detector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_loader.py
│   ├── nlp_engine.py
│   └── utils.py
│
└── app.py

🚀 How to Run
pip install -r requirements.txt
python data_loader.py
python nlp_engine.py
python app.py

🔮 Future Work

Expand query aliases for topic-level entities

Scale comment volume beyond per-entity cap

Cross-market comparison (US vs AU tech narratives)