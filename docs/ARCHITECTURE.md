# System Architecture – AI Bubble Detector

## 1. Overview
This project implements an end-to-end NLP-driven analytics pipeline that transforms unstructured public discussions from Hacker News into structured, interpretable market signals such as Hype, Technical Moat, and Bubble Index.

The system supports both:
- **Company-level analysis** (e.g., OpenAI, Google Gemini)
- **Topic-level analysis** (e.g., DeepSeek, AI Agents)

This design ensures sufficient discussion volume while enabling comparative analysis across different types of market narratives.

---

## 2. End-to-End Data Flow

Hacker News (Algolia API)
        ↓
Data Acquisition & Cleaning (`data_loader.py`)
        ↓
Raw Comment Dataset (`raw_hn_data.csv`)
        ↓
Zero-shot NLP Scoring + Sentiment (`nlp_engine.py`)
        ↓
Structured Metrics Dataset (`processed_scored_data.csv`)
        ↓
Aggregation & Visualization (`app.py`)
        ↓
Dashboard + BI Export (`mock_data.csv`)

---

## 3. Data Contracts (Key Fields)

### Raw Data (data/raw)
- `comment_id`
- `created_at`
- `entity`
- `entity_type` (company / topic)
- `text`
- `url`

### Processed Data (data/processed)
- `date`
- `entity`
- `hype_score`
- `moat_score`
- `sentiment_score`
- `bubble_index`

---

## 4. Module Responsibilities

### `src/data_loader.py`
**Purpose:**  
Fetch and clean Hacker News comments using Algolia API.

**Responsibilities:**
- Query Hacker News using entity-specific keywords
- Support both company-level and topic-level entities
- Enforce per-entity comment cap (default: 100) for rapid iteration
- Remove empty or deleted comments
- Deduplicate records by comment ID
- Save raw structured dataset

**Output:** `raw_hn_data.csv`

---

### `src/nlp_engine.py`
**Purpose:**  
Convert unstructured text into structured analytical features.

**Responsibilities:**
- Zero-shot classification using `facebook/bart-large-mnli`
- Candidate labels:
  - Technical Moat
  - Marketing Hype
  - Monetization Strategy
  - Real World Usage
- Sentiment scoring (VADER / TextBlob)
- Compute composite Bubble Index

**Output:** `processed_scored_data.csv`

---

### `app.py`
**Purpose:**  
Visualization, exploration, and BI export.

**Responsibilities:**
- Load processed metrics
- Filter by entity
- Aggregate metrics by date
- Render Streamlit dashboard
- Export BI-ready CSV for PowerBI

---

## 5. Key Design Decisions

1. **Algolia API over HTML scraping**  
   Chosen for stability, speed, and consistent schema.

2. **Zero-shot learning instead of supervised models**  
   Avoids the need for labeled data while enabling semantic classification.

3. **Entity abstraction (company vs topic)**  
   Improves signal density and allows narrative-level analysis.

4. **Daily aggregation for stability**  
   Reduces noise and improves temporal consistency.

---

## 6. Scope & Limitations
- Measures public discussion structure, not internal technical quality.
- Results reflect perception-driven signals.
- Sparse entities may require additional query expansion or topic abstraction.
