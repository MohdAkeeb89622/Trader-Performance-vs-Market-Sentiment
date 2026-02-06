# Trader Performance vs Market Sentiment Analysis

## Executive Summary

This analysis examines the relationship between Bitcoin market sentiment (Fear/Greed Index) and trader behavior/performance on Hyperliquid. The study covers **211,224 trades** from **32 unique traders** between March 2023 and June 2025.

---

## Methodology

### Data Sources
1. **Bitcoin Fear/Greed Index** (2,644 daily observations from 2018-2025)
2. **Hyperliquid Historical Trades** (211,224 trades with PnL, position data)

### Analysis Approach
- Aligned datasets by date (daily level)
- Created key metrics: daily PnL, win rate, trade frequency, long/short ratio
- Segmented traders by volume, frequency, and performance consistency
- Compared metrics across Fear vs Greed market conditions

---

## Key Insights

### Insight 1: Counter-Intuitive Performance Pattern
**Traders perform significantly BETTER during Fear periods than Greed periods.**

| Metric | Fear Days | Greed Days |
|--------|-----------|------------|
| Avg Daily PnL | $209,373 | $78,341 |
| Win Rate | 41.6% | 35.0% |
| Negative Days | 6.3% | 15.6% |

*Implication: Market fear creates profitable opportunities for active traders.*

### Insight 2: Trading Activity Peaks During Fear
Traders are **3.7x more active** during Fear periods (4,183 trades/day vs 1,120 during Greed), suggesting experienced traders capitalize on volatility.

### Insight 3: High-Volume Traders Excel in Fear Markets
High-volume traders achieve **$340,703 avg PnL on Fear days** vs **$102,464 on Greed days**—a 3.3x performance difference. Larger players are better positioned to exploit fear-driven price dislocations.

---

## Strategy Recommendations

### Strategy 1: Contrarian Volume Approach
**During Fear periods (Index < 40):**
- Increase position sizes by 20-30%
- High-volume traders should be most aggressive
- Target fear-driven price dislocations

**Rationale:** Data shows traders earn 2.7x more PnL during Fear vs Greed.

### Strategy 2: Sentiment-Adaptive Trading
**During Fear:**
- Focus on fewer, higher-conviction trades
- Maintain short bias (long ratio drops to 46%)

**During Greed:**
- Reduce position sizes
- Set tighter stop-losses (reversal risk higher)
- Maximum 60% long exposure

---

## Conclusions

1. **Market sentiment is a strong predictor of trader success** - Fear days consistently outperform Greed days
2. **Volume matters** - High-volume traders capture the most value during volatility
3. **Frequency ≠ Profitability** - More trades during fear, but focus on quality over quantity
4. **Contrarian approach works** - Going against market sentiment yields better results

---

*Analysis completed: February 2026*  
*Charts and detailed tables available in `/outputs/` directory*
