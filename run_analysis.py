#!/usr/bin/env python3
"""
Trader Performance vs Market Sentiment Analysis
================================================
Complete analysis script for PrimeTrade Data Science Intern Assignment

This script generates:
- All required charts (saved to outputs/charts/)
- All required tables (saved to outputs/tables/)
- Analysis metrics and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
CHARTS_DIR = BASE_DIR / 'outputs' / 'charts'
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'

# Ensure output directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRADER PERFORMANCE VS MARKET SENTIMENT ANALYSIS")
print("=" * 70)
print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# PART A: DATA PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("PART A: DATA PREPARATION")
print("=" * 70)

# Load datasets
print("\n📂 Loading datasets...")
trades = pd.read_csv(DATA_DIR / 'historical_data.csv')
sentiment = pd.read_csv(DATA_DIR / 'fear_greed_index.csv')

# Document dataset info
print("\n📊 DATASET DOCUMENTATION:")
print("-" * 50)
print(f"\n1. TRADES DATA (historical_data.csv):")
print(f"   • Rows: {trades.shape[0]:,}")
print(f"   • Columns: {trades.shape[1]}")
print(f"   • Columns list: {trades.columns.tolist()}")

print(f"\n2. SENTIMENT DATA (fear_greed_index.csv):")
print(f"   • Rows: {sentiment.shape[0]:,}")
print(f"   • Columns: {sentiment.shape[1]}")
print(f"   • Columns list: {sentiment.columns.tolist()}")

# Missing values analysis
print("\n📋 MISSING VALUES ANALYSIS:")
print("-" * 50)
print("\nTrades data missing values:")
trades_missing = trades.isnull().sum()
trades_missing_pct = (trades_missing / len(trades) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': trades_missing,
    'Missing %': trades_missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0].to_string() if missing_df['Missing Count'].sum() > 0 else "   No missing values found!")

print("\nSentiment data missing values:")
sentiment_missing = sentiment.isnull().sum()
print(f"   Total missing values: {sentiment_missing.sum()}")

# Duplicate detection
print("\n🔍 DUPLICATE DETECTION:")
print("-" * 50)
trades_duplicates = trades.duplicated().sum()
sentiment_duplicates = sentiment.duplicated().sum()
print(f"   Trades data duplicates: {trades_duplicates:,}")
print(f"   Sentiment data duplicates: {sentiment_duplicates:,}")

# Save data quality report
data_quality = pd.DataFrame({
    'Dataset': ['trades', 'sentiment'],
    'Rows': [trades.shape[0], sentiment.shape[0]],
    'Columns': [trades.shape[1], sentiment.shape[1]],
    'Missing Values': [trades.isnull().sum().sum(), sentiment.isnull().sum().sum()],
    'Duplicates': [trades_duplicates, sentiment_duplicates]
})
data_quality.to_csv(TABLES_DIR / 'data_quality_report.csv', index=False)

# Convert timestamps and align datasets
print("\n⏰ TIMESTAMP CONVERSION:")
print("-" * 50)

# Trades: Use 'Timestamp' column (Unix timestamp)
trades['date'] = pd.to_datetime(trades['Timestamp'], unit='ms', utc=True).dt.date
trades['date'] = pd.to_datetime(trades['date'])
print(f"   Trades date range: {trades['date'].min()} to {trades['date'].max()}")

# Sentiment: Use 'date' column
sentiment['date'] = pd.to_datetime(sentiment['date'])
print(f"   Sentiment date range: {sentiment['date'].min()} to {sentiment['date'].max()}")

# Create simplified sentiment mapping (Fear includes Extreme Fear, Greed includes Extreme Greed)
sentiment['sentiment'] = sentiment['classification'].apply(
    lambda x: 'Fear' if 'Fear' in str(x) else 'Greed'
)

# Feature Engineering
print("\n🔧 FEATURE ENGINEERING:")
print("-" * 50)

# Identify key columns
ACCOUNT_COL = 'Account'
PNL_COL = 'Closed PnL'
SIZE_COL = 'Size USD'
SIDE_COL = 'Side'

# Create derived features
trades['is_win'] = trades[PNL_COL] > 0
trades['is_long'] = trades[SIDE_COL].str.upper() == 'BUY'
trades['trade_size'] = trades[SIZE_COL].abs()

print(f"   ✓ Created 'is_win' flag (profitable trades)")
print(f"   ✓ Created 'is_long' flag (long positions)")
print(f"   ✓ Created 'trade_size' (absolute trade size)")

# Daily aggregation per trader
print("\n📈 DAILY AGGREGATION:")
print("-" * 50)

daily = trades.groupby([ACCOUNT_COL, 'date']).agg(
    daily_pnl=(PNL_COL, 'sum'),
    win_rate=('is_win', 'mean'),
    trades_per_day=(PNL_COL, 'count'),
    avg_trade_size=('trade_size', 'mean'),
    total_volume=('trade_size', 'sum'),
    long_ratio=('is_long', 'mean')
).reset_index()

print(f"   Daily aggregated records: {len(daily):,}")

# Merge with sentiment
daily = daily.merge(
    sentiment[['date', 'sentiment', 'value', 'classification']],
    on='date',
    how='left'
)
daily = daily.dropna(subset=['sentiment'])
print(f"   Records after sentiment merge: {len(daily):,}")

# Save daily metrics
daily.to_csv(TABLES_DIR / 'daily_trader_metrics.csv', index=False)

# =============================================================================
# PART B: ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("PART B: ANALYSIS")
print("=" * 70)

# -----------------------------------------------------------------------------
# Question 1: Performance differences between Fear vs Greed days
# -----------------------------------------------------------------------------
print("\n📊 Q1: PERFORMANCE DIFFERENCE (FEAR VS GREED)")
print("-" * 50)

performance_by_sentiment = daily.groupby('sentiment').agg(
    avg_daily_pnl=('daily_pnl', 'mean'),
    median_daily_pnl=('daily_pnl', 'median'),
    avg_win_rate=('win_rate', 'mean'),
    total_pnl=('daily_pnl', 'sum'),
    trader_days=('daily_pnl', 'count'),
    pnl_std=('daily_pnl', 'std')
).round(4)

# Calculate drawdown proxy (negative PnL days)
fear_data = daily[daily['sentiment'] == 'Fear']['daily_pnl']
greed_data = daily[daily['sentiment'] == 'Greed']['daily_pnl']

performance_by_sentiment['pct_negative_days'] = daily.groupby('sentiment').apply(
    lambda x: (x['daily_pnl'] < 0).mean()
).values
performance_by_sentiment['max_drawdown'] = daily.groupby('sentiment')['daily_pnl'].min().values

print(performance_by_sentiment.to_string())
performance_by_sentiment.to_csv(TABLES_DIR / 'performance_by_sentiment.csv')

# Statistical comparison
print(f"\n   📌 KEY FINDINGS:")
print(f"   • Average PnL on Fear days: ${performance_by_sentiment.loc['Fear', 'avg_daily_pnl']:.2f}")
print(f"   • Average PnL on Greed days: ${performance_by_sentiment.loc['Greed', 'avg_daily_pnl']:.2f}")
print(f"   • Win rate on Fear days: {performance_by_sentiment.loc['Fear', 'avg_win_rate']*100:.1f}%")
print(f"   • Win rate on Greed days: {performance_by_sentiment.loc['Greed', 'avg_win_rate']*100:.1f}%")

# Chart 1: PnL Distribution by Sentiment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=daily, x='sentiment', y='daily_pnl', ax=axes[0], palette=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Daily PnL Distribution by Market Sentiment', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Market Sentiment', fontsize=12)
axes[0].set_ylabel('Daily PnL ($)', fontsize=12)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Bar chart for metrics
metrics = ['avg_daily_pnl', 'avg_win_rate']
x = np.arange(len(metrics))
width = 0.35

fear_vals = [performance_by_sentiment.loc['Fear', 'avg_daily_pnl'], 
             performance_by_sentiment.loc['Fear', 'avg_win_rate'] * 100]
greed_vals = [performance_by_sentiment.loc['Greed', 'avg_daily_pnl'], 
              performance_by_sentiment.loc['Greed', 'avg_win_rate'] * 100]

bars1 = axes[1].bar(x - width/2, fear_vals, width, label='Fear', color='#FF6B6B')
bars2 = axes[1].bar(x + width/2, greed_vals, width, label='Greed', color='#4ECDC4')

axes[1].set_title('Performance Metrics: Fear vs Greed', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Avg Daily PnL ($)', 'Win Rate (%)'])
axes[1].legend()
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(CHARTS_DIR / 'chart1_pnl_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n   ✓ Saved: chart1_pnl_by_sentiment.png")

# -----------------------------------------------------------------------------
# Question 2: Behavior changes based on sentiment
# -----------------------------------------------------------------------------
print("\n📊 Q2: BEHAVIOR CHANGES BY SENTIMENT")
print("-" * 50)

behavior_by_sentiment = daily.groupby('sentiment').agg(
    avg_trades_per_day=('trades_per_day', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    avg_long_ratio=('long_ratio', 'mean'),
    total_volume=('total_volume', 'sum'),
    unique_traders=(ACCOUNT_COL, 'nunique')
).round(4)

print(behavior_by_sentiment.to_string())
behavior_by_sentiment.to_csv(TABLES_DIR / 'behavior_by_sentiment.csv')

print(f"\n   📌 KEY FINDINGS:")
print(f"   • Avg trades/day on Fear: {behavior_by_sentiment.loc['Fear', 'avg_trades_per_day']:.1f}")
print(f"   • Avg trades/day on Greed: {behavior_by_sentiment.loc['Greed', 'avg_trades_per_day']:.1f}")
print(f"   • Long ratio on Fear: {behavior_by_sentiment.loc['Fear', 'avg_long_ratio']*100:.1f}%")
print(f"   • Long ratio on Greed: {behavior_by_sentiment.loc['Greed', 'avg_long_ratio']*100:.1f}%")

# Chart 2: Behavior Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trade frequency
ax1 = axes[0, 0]
sentiment_order = ['Fear', 'Greed']
colors = ['#FF6B6B', '#4ECDC4']
sns.barplot(data=daily, x='sentiment', y='trades_per_day', ax=ax1, 
            order=sentiment_order, palette=colors, estimator=np.mean)
ax1.set_title('Average Trades per Day', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Trades per Day')

# Trade size
ax2 = axes[0, 1]
sns.barplot(data=daily, x='sentiment', y='avg_trade_size', ax=ax2, 
            order=sentiment_order, palette=colors, estimator=np.mean)
ax2.set_title('Average Trade Size ($)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Trade Size ($)')

# Long/Short ratio
ax3 = axes[1, 0]
sns.barplot(data=daily, x='sentiment', y='long_ratio', ax=ax3, 
            order=sentiment_order, palette=colors, estimator=np.mean)
ax3.set_title('Long Position Ratio', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sentiment')
ax3.set_ylabel('Long Ratio')
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Neutral (50%)')
ax3.legend()

# Volume distribution
ax4 = axes[1, 1]
volume_by_sentiment = daily.groupby('sentiment')['total_volume'].sum()
ax4.pie(volume_by_sentiment, labels=volume_by_sentiment.index, autopct='%1.1f%%',
        colors=colors, explode=[0.02, 0.02])
ax4.set_title('Total Volume Distribution', fontsize=12, fontweight='bold')

plt.suptitle('Trader Behavior by Market Sentiment', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'chart2_behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: chart2_behavior_by_sentiment.png")

# -----------------------------------------------------------------------------
# Question 3: Trader Segmentation
# -----------------------------------------------------------------------------
print("\n📊 Q3: TRADER SEGMENTATION")
print("-" * 50)

# Aggregate at trader level
trader_stats = trades.groupby(ACCOUNT_COL).agg(
    total_pnl=(PNL_COL, 'sum'),
    total_trades=(PNL_COL, 'count'),
    win_rate=('is_win', 'mean'),
    avg_trade_size=('trade_size', 'mean'),
    trading_days=('date', 'nunique'),
    total_volume=('trade_size', 'sum')
).reset_index()

trader_stats['trades_per_day'] = trader_stats['total_trades'] / trader_stats['trading_days']

# Segment 1: High vs Low Volume Traders
volume_median = trader_stats['total_volume'].median()
trader_stats['volume_segment'] = np.where(
    trader_stats['total_volume'] >= volume_median, 'High Volume', 'Low Volume'
)

# Segment 2: Frequent vs Infrequent Traders
trades_median = trader_stats['trades_per_day'].median()
trader_stats['frequency_segment'] = np.where(
    trader_stats['trades_per_day'] >= trades_median, 'Frequent', 'Infrequent'
)

# Segment 3: Consistent Winners vs Inconsistent
win_rate_threshold = 0.5
pnl_positive = trader_stats['total_pnl'] > 0
high_win_rate = trader_stats['win_rate'] >= win_rate_threshold
trader_stats['performance_segment'] = 'Inconsistent'
trader_stats.loc[pnl_positive & high_win_rate, 'performance_segment'] = 'Consistent Winner'
trader_stats.loc[~pnl_positive & ~high_win_rate, 'performance_segment'] = 'Consistent Loser'

# Segment summary
print("\n   SEGMENT 1: VOLUME-BASED")
volume_summary = trader_stats.groupby('volume_segment').agg(
    count=('total_pnl', 'count'),
    avg_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
).round(2)
print(volume_summary.to_string())

print("\n   SEGMENT 2: FREQUENCY-BASED")
freq_summary = trader_stats.groupby('frequency_segment').agg(
    count=('total_pnl', 'count'),
    avg_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
).round(2)
print(freq_summary.to_string())

print("\n   SEGMENT 3: PERFORMANCE-BASED")
perf_summary = trader_stats.groupby('performance_segment').agg(
    count=('total_pnl', 'count'),
    avg_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
).round(2)
print(perf_summary.to_string())

# Save segmentation data
trader_stats.to_csv(TABLES_DIR / 'trader_segments.csv', index=False)
volume_summary.to_csv(TABLES_DIR / 'segment_volume_summary.csv')
freq_summary.to_csv(TABLES_DIR / 'segment_frequency_summary.csv')
perf_summary.to_csv(TABLES_DIR / 'segment_performance_summary.csv')

# Chart 3: Trader Segmentation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Volume Segment Performance
ax1 = axes[0, 0]
sns.barplot(data=trader_stats, x='volume_segment', y='total_pnl', ax=ax1,
            palette=['#3498db', '#e74c3c'], estimator=np.mean)
ax1.set_title('Avg PnL: High vs Low Volume Traders', fontsize=12, fontweight='bold')
ax1.set_xlabel('Volume Segment')
ax1.set_ylabel('Average Total PnL ($)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Frequency Segment Performance
ax2 = axes[0, 1]
sns.barplot(data=trader_stats, x='frequency_segment', y='total_pnl', ax=ax2,
            palette=['#9b59b6', '#1abc9c'], estimator=np.mean)
ax2.set_title('Avg PnL: Frequent vs Infrequent Traders', fontsize=12, fontweight='bold')
ax2.set_xlabel('Frequency Segment')
ax2.set_ylabel('Average Total PnL ($)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Performance Segment Distribution
ax3 = axes[1, 0]
perf_counts = trader_stats['performance_segment'].value_counts()
colors_perf = ['#2ecc71', '#e74c3c', '#f39c12']
ax3.pie(perf_counts, labels=perf_counts.index, autopct='%1.1f%%',
        colors=colors_perf[:len(perf_counts)], explode=[0.02]*len(perf_counts))
ax3.set_title('Trader Performance Distribution', fontsize=12, fontweight='bold')

# Win Rate by Segment
ax4 = axes[1, 1]
segment_winrate = trader_stats.groupby('performance_segment')['win_rate'].mean().sort_values()
segment_winrate.plot(kind='barh', ax=ax4, color=['#e74c3c', '#f39c12', '#2ecc71'][:len(segment_winrate)])
ax4.set_title('Win Rate by Performance Segment', fontsize=12, fontweight='bold')
ax4.set_xlabel('Win Rate')
ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='50% threshold')

plt.suptitle('Trader Segmentation Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'chart3_trader_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n   ✓ Saved: chart3_trader_segmentation.png")

# -----------------------------------------------------------------------------
# Question 4: Key Insights with Additional Charts
# -----------------------------------------------------------------------------
print("\n📊 Q4: KEY INSIGHTS")
print("-" * 50)

# Merge trader segments with daily data
daily_with_segments = daily.merge(
    trader_stats[[ACCOUNT_COL, 'volume_segment', 'frequency_segment', 'performance_segment']],
    on=ACCOUNT_COL,
    how='left'
)

# Insight 1: Segment performance during different sentiments
segment_sentiment = daily_with_segments.groupby(['volume_segment', 'sentiment']).agg(
    avg_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
).reset_index()

# Chart 4: Segment Performance by Sentiment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# High Volume vs Low Volume by Sentiment
ax1 = axes[0]
pivot_data = segment_sentiment.pivot(index='volume_segment', columns='sentiment', values='avg_pnl')
pivot_data.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Avg Daily PnL by Volume Segment & Sentiment', fontsize=12, fontweight='bold')
ax1.set_xlabel('Volume Segment')
ax1.set_ylabel('Avg Daily PnL ($)')
ax1.legend(title='Sentiment')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

# Time series of sentiment and PnL
ax2 = axes[1]
daily_agg = daily.groupby('date').agg(
    avg_pnl=('daily_pnl', 'mean'),
    sentiment_value=('value', 'first')
).reset_index()

ax2_twin = ax2.twinx()
ax2.plot(daily_agg['date'], daily_agg['avg_pnl'].rolling(7).mean(), 
         color='#3498db', linewidth=2, label='Avg PnL (7-day MA)')
ax2_twin.fill_between(daily_agg['date'], daily_agg['sentiment_value'], 
                       alpha=0.3, color='#e74c3c', label='Fear/Greed Index')
ax2.set_title('PnL Trend vs Fear/Greed Index', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Avg Daily PnL ($)', color='#3498db')
ax2_twin.set_ylabel('Fear/Greed Index', color='#e74c3c')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=50, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / 'chart4_insights_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: chart4_insights_visualization.png")

# Chart 5: Comprehensive Dashboard
fig = plt.figure(figsize=(16, 12))

# Performance metrics heatmap
ax1 = fig.add_subplot(2, 2, 1)
metrics_matrix = daily.groupby('sentiment').agg({
    'daily_pnl': 'mean',
    'win_rate': 'mean',
    'trades_per_day': 'mean',
    'avg_trade_size': 'mean',
    'long_ratio': 'mean'
}).T
sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax1)
ax1.set_title('Performance Metrics Heatmap', fontsize=12, fontweight='bold')

# PnL distribution histogram
ax2 = fig.add_subplot(2, 2, 2)
for sent, color in zip(['Fear', 'Greed'], ['#FF6B6B', '#4ECDC4']):
    data = daily[daily['sentiment'] == sent]['daily_pnl']
    ax2.hist(data, bins=50, alpha=0.6, color=color, label=sent, density=True)
ax2.set_title('PnL Distribution by Sentiment', fontsize=12, fontweight='bold')
ax2.set_xlabel('Daily PnL ($)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)

# Top performers analysis
ax3 = fig.add_subplot(2, 2, 3)
top_traders = trader_stats.nlargest(10, 'total_pnl')
ax3.barh(range(len(top_traders)), top_traders['total_pnl'], color='#2ecc71')
ax3.set_yticks(range(len(top_traders)))
ax3.set_yticklabels([f"Trader {i+1}" for i in range(len(top_traders))])
ax3.set_title('Top 10 Traders by Total PnL', fontsize=12, fontweight='bold')
ax3.set_xlabel('Total PnL ($)')

# Trading activity by day of week
ax4 = fig.add_subplot(2, 2, 4)
daily['day_of_week'] = pd.to_datetime(daily['date']).dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_pnl = daily.groupby('day_of_week')['daily_pnl'].mean().reindex(day_order)
colors_days = ['#3498db' if pnl >= 0 else '#e74c3c' for pnl in day_pnl]
ax4.bar(range(len(day_pnl)), day_pnl.values, color=colors_days)
ax4.set_xticks(range(len(day_pnl)))
ax4.set_xticklabels([d[:3] for d in day_order], rotation=45)
ax4.set_title('Average PnL by Day of Week', fontsize=12, fontweight='bold')
ax4.set_ylabel('Avg Daily PnL ($)')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

plt.suptitle('Trader Performance Dashboard', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'chart5_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: chart5_comprehensive_dashboard.png")

# =============================================================================
# PART C: ACTIONABLE OUTPUT - Strategy Recommendations
# =============================================================================
print("\n" + "=" * 70)
print("PART C: ACTIONABLE OUTPUT - STRATEGY RECOMMENDATIONS")
print("=" * 70)

# Calculate key metrics for strategy recommendations
fear_pnl = performance_by_sentiment.loc['Fear', 'avg_daily_pnl']
greed_pnl = performance_by_sentiment.loc['Greed', 'avg_daily_pnl']
fear_winrate = performance_by_sentiment.loc['Fear', 'avg_win_rate']
greed_winrate = performance_by_sentiment.loc['Greed', 'avg_win_rate']

# Segment-specific metrics
high_vol_fear = daily_with_segments[(daily_with_segments['volume_segment'] == 'High Volume') & 
                                      (daily_with_segments['sentiment'] == 'Fear')]['daily_pnl'].mean()
high_vol_greed = daily_with_segments[(daily_with_segments['volume_segment'] == 'High Volume') & 
                                       (daily_with_segments['sentiment'] == 'Greed')]['daily_pnl'].mean()
low_vol_fear = daily_with_segments[(daily_with_segments['volume_segment'] == 'Low Volume') & 
                                     (daily_with_segments['sentiment'] == 'Fear')]['daily_pnl'].mean()
low_vol_greed = daily_with_segments[(daily_with_segments['volume_segment'] == 'Low Volume') & 
                                      (daily_with_segments['sentiment'] == 'Greed')]['daily_pnl'].mean()

strategies = []

# Strategy 1: Volume-based sentiment strategy
if high_vol_fear > high_vol_greed:
    strat1 = f"""
STRATEGY 1: CONTRARIAN VOLUME APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation: High-volume traders perform BETTER during Fear days 
  (Avg PnL: ${high_vol_fear:.2f}) vs Greed days (${high_vol_greed:.2f})

Recommendation: 
  • INCREASE position sizes during Fear periods (when index < 40)
  • REDUCE position sizes during Greed periods (when index > 60)
  • High-volume traders should be more aggressive during market fear
  
Rationale: Counter-intuitive result suggests high-volume traders 
capitalize on fear-driven volatility and discounted entry points.
"""
else:
    strat1 = f"""
STRATEGY 1: MOMENTUM VOLUME APPROACH  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation: High-volume traders perform BETTER during Greed days 
  (Avg PnL: ${high_vol_greed:.2f}) vs Fear days (${high_vol_fear:.2f})

Recommendation:
  • INCREASE position sizes during Greed periods (when index > 60)
  • REDUCE position sizes during Fear periods (when index < 40)
  • Ride the momentum - larger positions when market is bullish
  
Rationale: High-volume traders benefit from momentum-driven markets
where bullish sentiment creates favorable trading conditions.
"""
strategies.append(strat1)
print(strat1)

# Strategy 2: Frequency-based strategy
freq_fear = daily_with_segments[daily_with_segments['sentiment'] == 'Fear'].groupby('frequency_segment')['daily_pnl'].mean()
freq_greed = daily_with_segments[daily_with_segments['sentiment'] == 'Greed'].groupby('frequency_segment')['daily_pnl'].mean()

# Long ratio analysis
fear_long_ratio = behavior_by_sentiment.loc['Fear', 'avg_long_ratio']
greed_long_ratio = behavior_by_sentiment.loc['Greed', 'avg_long_ratio']

strat2 = f"""
STRATEGY 2: SENTIMENT-ADAPTIVE TRADING FREQUENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation: 
  • Long ratio during Fear: {fear_long_ratio*100:.1f}%
  • Long ratio during Greed: {greed_long_ratio*100:.1f}%
  • Trade frequency changes across sentiment regimes

Recommendation:
  • During FEAR days: 
    - Reduce trading frequency by 20-30%
    - Focus on higher-conviction trades only
    - Consider increasing short exposure if long ratio > 55%
    
  • During GREED days:
    - Maintain normal trading frequency
    - Slight long bias acceptable (up to 60%)
    - Set tighter stop-losses due to potential reversals
    
Rationale: Fear periods show higher volatility and less predictable
outcomes. Reducing frequency during fear preserves capital while 
maintaining exposure to high-probability setups.
"""
strategies.append(strat2)
print(strat2)

# Save strategies
with open(TABLES_DIR / 'strategy_recommendations.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ACTIONABLE STRATEGY RECOMMENDATIONS\n")
    f.write("Based on Trader Performance vs Market Sentiment Analysis\n")
    f.write("=" * 70 + "\n\n")
    for strat in strategies:
        f.write(strat + "\n")

print(f"\n   ✓ Saved: strategy_recommendations.txt")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

summary_stats = {
    'Total Trades Analyzed': len(trades),
    'Unique Traders': trades[ACCOUNT_COL].nunique(),
    'Date Range': f"{trades['date'].min().strftime('%Y-%m-%d')} to {trades['date'].max().strftime('%Y-%m-%d')}",
    'Fear Days': len(sentiment[sentiment['sentiment'] == 'Fear']),
    'Greed Days': len(sentiment[sentiment['sentiment'] == 'Greed']),
    'Avg PnL (Fear)': f"${fear_pnl:.2f}",
    'Avg PnL (Greed)': f"${greed_pnl:.2f}",
    'Win Rate (Fear)': f"{fear_winrate*100:.1f}%",
    'Win Rate (Greed)': f"{greed_winrate*100:.1f}%",
}

for key, value in summary_stats.items():
    print(f"   {key}: {value}")

# Save summary stats
pd.DataFrame([summary_stats]).T.to_csv(TABLES_DIR / 'analysis_summary.csv', header=['Value'])

print("\n" + "=" * 70)
print("OUTPUT FILES GENERATED")
print("=" * 70)

print("\n📊 CHARTS (outputs/charts/):")
for chart in sorted(CHARTS_DIR.glob('*.png')):
    print(f"   • {chart.name}")

print("\n📋 TABLES (outputs/tables/):")
for table in sorted(TABLES_DIR.glob('*')):
    if table.name != '.gitkeep':
        print(f"   • {table.name}")

print("\n" + "=" * 70)
print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
