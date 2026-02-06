# Trader Sentiment Analysis

**Trader Performance vs Market Sentiment (Fear/Greed)**

A comprehensive analysis of how Bitcoin market sentiment relates to trader behavior and performance on Hyperliquid.

---

## 🎯 Project Overview

This project analyzes 211,224+ trades from 32 unique traders to uncover patterns between market sentiment and trading outcomes. Key findings include:

- **2.7x higher PnL** during Fear periods vs Greed periods
- **41.6% win rate** on Fear days vs 35% on Greed days
- High-volume traders outperform by **3.3x** during market fear

---

## 📁 Project Structure

```
primetrade-trader-sentiment-analysis/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_analysis.py          # Main analysis script
│
├── notebook/
│   └── trader_sentiment_analysis.ipynb
│
├── data/
│   ├── historical_data.csv   # Hyperliquid trades (211K+ records)
│   └── fear_greed_index.csv  # Bitcoin Fear/Greed Index
│
├── outputs/
│   ├── charts/               # 5 visualization charts
│   │   ├── chart1_pnl_by_sentiment.png
│   │   ├── chart2_behavior_by_sentiment.png
│   │   ├── chart3_trader_segmentation.png
│   │   ├── chart4_insights_visualization.png
│   │   └── chart5_comprehensive_dashboard.png
│   │
│   └── tables/               # 10 analysis tables
│       ├── analysis_summary.csv
│       ├── behavior_by_sentiment.csv
│       ├── data_quality_report.csv
│       ├── daily_trader_metrics.csv
│       ├── performance_by_sentiment.csv
│       ├── segment_*.csv
│       ├── strategy_recommendations.txt
│       └── trader_segments.csv
│
└── writeup/
    └── summary.md            # Analysis summary & recommendations
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repo-url>
cd primetrade-trader-sentiment-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run the complete analysis
python run_analysis.py
```

This will:
- Load and validate both datasets
- Perform all required analysis (Parts A, B, C)
- Generate 5 charts in `outputs/charts/`
- Export 10 tables in `outputs/tables/`
- Print summary findings to console

### 3. View Results

- **Charts**: Open `outputs/charts/*.png`
- **Tables**: Open `outputs/tables/*.csv` in Excel/Sheets
- **Summary**: Read `writeup/summary.md`
- **Strategies**: See `outputs/tables/strategy_recommendations.txt`

---

## 📊 Key Deliverables

### Part A: Data Preparation ✅
- Dataset documentation (rows, columns, missing values)
- Timestamp alignment
- Key metrics: daily PnL, win rate, trade size, leverage, long/short ratio

### Part B: Analysis ✅
1. **Performance Comparison**: Fear vs Greed days
2. **Behavior Changes**: Trading frequency, position sizes, bias
3. **Trader Segmentation**: 3 segments (volume, frequency, consistency)
4. **Insights**: 3+ findings with supporting charts/tables

### Part C: Actionable Output ✅
- **Strategy 1**: Contrarian Volume Approach
- **Strategy 2**: Sentiment-Adaptive Trading Frequency

---

## 📈 Key Findings

| Metric | Fear Days | Greed Days | Insight |
|--------|-----------|------------|---------|
| Avg PnL | $209,373 | $78,341 | 2.7x better during Fear |
| Win Rate | 41.6% | 35.0% | Higher accuracy in Fear |
| Trades/Day | 4,183 | 1,120 | 3.7x more activity |
| Long Ratio | 45.9% | 49.4% | Slight short bias in Fear |

---

## 🛠 Technologies Used

- **Python 3.12+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Visualization
- **seaborn** - Statistical graphics
- **scikit-learn** - Machine learning (bonus model)

---

## 📝 Author

Data Science Intern Assignment for Primetrade.ai

---

## 📄 License

This project is for assessment purposes.
