# Research Design Plan: Momentum-Based Trading Strategy for Vietnam Markets

## 1. Project Overview

**Objective:** Implement and validate a momentum-based relative strength trading strategy adapted for Vietnam's long-only market constraints, testing whether past winners continue outperforming in an emerging market context.

**Research Questions:**
- Can momentum effects generate positive alpha in Vietnam's emerging market despite the inability to short losers?
- What is the optimal formation/holding period combination for Vietnam markets?
- How do momentum profits decompose between systematic risk compensation and behavioral factors?

**Scope:** 
- **Markets:** Vietnam equity markets (HOSE & HNX listed stocks)
- **Time Period:** January 2010 - August 2025 (15.5 years)
- **Strategy Type:** Long-only momentum with benchmark comparison
- **Universe:** All stocks with minimum 6-month trading history and daily volume > 100M VND

**Note on Data Requirements:** Following Jegadeesh & Titman (1993), the core momentum strategy requires ONLY price/return data. The original paper used CRSP daily returns (already adjusted for corporate actions) for the main strategy. Earnings announcement data was used only for a supplementary event study (Section VIII) to understand profit sources, not for strategy implementation.

---

## 2. Data Design

### 2.1 Data Sources

**Primary Source:**
- **Price Data:** Historical OHLCV data from CSV files
  - Source: Vietnam stock exchanges (HOSE/HNX) historical data
  - Format: Daily OHLCV with adjusted closing prices
  - Structure: `date, ticker, open, high, low, close, volume`
  - Note: Closing prices are pre-adjusted for corporate actions (splits, dividends)
- **Market Index Data:** VN-Index, HNX-Index daily values from CSV
- **Volume Data:** Daily trading volume and value (already in OHLCV)

**Data Frequency:** 
- Daily OHLCV for return calculations
- Monthly aggregation for portfolio formation

### 2.2 Data Cleaning Rules

**Quality Filters:**
- Remove stocks with > 15 non-trading days in formation period
- Exclude stocks with price < 1,000 VND (penny stocks)
- Filter by average daily trading value < 100M VND (price × volume)
- Remove newly listed stocks (< 6 months trading history)
- Eliminate stocks with zero or missing volume data
- Optional: If market cap data available separately, filter < 100B VND

**Data Validation:**
```python
# Since prices are pre-adjusted, validate data quality and consistency
def validate_ohlcv_data(df):
    # Check for missing values
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        assert df[col].isna().sum() == 0, f"Missing values in {col}"
    
    # Check for zero/negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        assert (df[col] > 0).all(), f"Non-positive values in {col}"
    
    # Check OHLC relationships
    assert (df['high'] >= df['low']).all(), "High < Low detected"
    assert (df['high'] >= df['open']).all(), "High < Open detected"
    assert (df['high'] >= df['close']).all(), "High < Close detected"
    
    # Check for extreme price jumps (potential data errors)
    daily_returns = df['close'].pct_change()
    extreme_moves = daily_returns.abs() > 0.5
    if extreme_moves.any():
        print(f"Warning: {extreme_moves.sum()} days with >50% price moves")
    
    return df
```

### 2.3 Feature Engineering Plan

**Core Features (from OHLCV data only):**
```python
# Formation period returns (J-month) - the ONLY required feature
momentum_score[t] = (price[t] / price[t-J*21]) - 1

# Skip period adjustment (1 week = 5 trading days)
formation_end = t - 5
ranking_return = calculate_return(t-J*21-5, t-5)

# Optional: Risk-adjusted momentum using price-based volatility
realized_vol = calculate_volatility(daily_returns, window=60)
risk_adj_momentum = momentum_score / realized_vol
```

**Supporting Features (all derived from OHLCV):**
- Rolling volatility from daily returns (20, 60, 120 days)
- Average daily volume from volume column (20-day MA)
- Average daily turnover (Volume × Close price)
- Price-based liquidity measures (Amihud illiquidity ratio)
- Market cap approximation (if shares outstanding available)

---

## 3. Hypothesis & Signal Design

### 3.1 Core Hypotheses

**H1: Momentum Continuation**
- *Hypothesis:* Winners (top decile) outperform market by 2-5% annually
- *Rationale:* Information diffusion is slower in emerging markets
- *Test:* Compare winner portfolio returns vs. VN-Index

**H2: Optimal Horizon**
- *Hypothesis:* 6-month/6-month strategy is optimal for Vietnam
- *Rationale:* Balance between signal strength and transaction costs
- *Test:* Grid search across J={3,6,9,12} × K={1,3,6,12}

**H3: Decomposition Sources**
- *Hypothesis:* Profits arise from both risk compensation and behavioral factors
- *Test:* Decompose returns using factor model:
  ```
  E(r_it - r_t)(r_it-1 - r_t-1) = σ²_μ + σ²_b * Cov(f_t, f_t-1) + Cov(e_it, e_it-1)
  ```

### 3.2 Signal Variables

**Primary Signal:**
- J-month cumulative return (skip-adjusted)

**Signal Enhancements (for robustness - all from price data):**
- Volume-weighted momentum (using volume column)
- Volatility-scaled momentum (using return volatility)
- 52-week high momentum (price relative to trailing high)

---

## 4. Modeling Approach

### 4.1 Baseline Model

**Simple Momentum Strategy:**
```python
def baseline_momentum(returns, J=6, K=6):
    # Rank stocks by J-month returns
    momentum_scores = calculate_momentum(returns, J, skip=5)
    deciles = pd.qcut(momentum_scores, 10, labels=False)
    
    # Form equal-weighted portfolios
    winner_portfolio = stocks[deciles == 9]
    
    # Hold for K months with monthly rebalancing
    return calculate_portfolio_returns(winner_portfolio, K)
```


### 4.3 Portfolio Construction

**Overlapping Portfolio Method:**
```python
# Maintain K overlapping portfolios
for month in range(K):
    # Each month: close 1/K old positions, open 1/K new
    portfolio_weight[stock] = sum(active_portfolios[stock]) / K
```

**Position Sizing:**
- Equal-weighted within deciles (baseline)
- Value-weighted (robustness check)
- Volatility-weighted (risk parity approach)

### 4.4 Cross-Validation Method

**Walk-Forward Analysis:**
- Training window: 36 months
- Validation window: 12 months
- Step size: 1 month
- Re-optimization frequency: Quarterly

---

## 5. Evaluation Framework

### 5.1 Backtesting Methodology

**Simulation Parameters:**
```python
backtest_config = {
    'initial_capital': 1_000_000_000,  # 1 billion VND
    'rebalance_frequency': 'monthly',
    'transaction_cost': 0.0025,  # 25 bps per side
    'slippage_model': 'linear',  # Based on daily volume
    'market_impact': 0.0010,  # 10 bps for large trades
}
```

**Data Split:**
- In-sample: 2010-2019 (strategy development)
- Out-of-sample: 2020-2025 (validation)
- Subperiod analysis: COVID crash, recovery periods

### 5.2 Performance Metrics

**Return Metrics:**
- Annualized return (CAGR)
- Excess return over VN-Index
- Information Ratio: `IR = (R_p - R_b) / TE`
- Sharpe Ratio: `SR = (R_p - R_f) / σ_p`

**Risk Metrics:**
- Maximum drawdown and duration
- Downside deviation
- Value at Risk (95%, 99%)
- Beta to VN-Index

**Implementation Metrics:**
- Average turnover: ~84.8% semi-annually
- Break-even transaction cost
- Capacity analysis (AUM limits)

### 5.3 Statistical Testing

**Significance Tests:**
```python
# Test for positive alpha
alpha_test = sm.OLS(excess_returns, market_returns).fit()
t_stat = alpha_test.params[0] / alpha_test.bse[0]

# Newey-West adjusted for autocorrelation
nw_se = NeweyWest(lags=6).fit(alpha_test)
```

**Robustness Checks:**
- Bootstrap confidence intervals (1000 iterations)
- Monte Carlo simulation for parameter sensitivity
- Regime analysis (bull/bear markets)

---

## 6. Risk & Constraint Analysis

### 6.1 Market-Specific Constraints

**Vietnam Market Rules:**
- **No short selling:** Long-only implementation required
- **Daily price limits:** ±7% for most stocks
- **T+2 settlement:** Affects rebalancing timing
- **Trading suspension risk:** Stocks can halt for news

### 6.2 Operational Risks

**Execution Risks:**
- **Liquidity constraints:** Filter stocks with ADV < 1B VND
- **Market impact:** Large orders move prices
- **Timing risk:** Intraday volatility affects execution

**Data Risks:**
- Survivorship bias in historical data
- Look-ahead bias in return calculations
- Quality of pre-adjusted prices from data source
- Missing data for thinly traded stocks

### 6.3 Strategy-Specific Risks

**Momentum Crashes:**
- Historical drawdowns during market reversals
- Implement dynamic hedging during high volatility
- Monitor correlation with market beta

**Capacity Constraints:**
```python
# Maximum strategy capacity
max_aum = median_daily_volume * 0.10 * num_stocks * 20
# Assume 10% of ADV, 20 trading days to build position
```

---

## 7. Experiment Tracking

### 7.1 Version Control

**Code Management:**
```yaml
repository: github.com/team/vietnam-momentum
branches:
  - main: Production-ready code
  - develop: Active development
  - experiments/: Feature testing
```

**Data Versioning:**
- DVC for large datasets
- MD5 checksums for data integrity
- Timestamp all data pulls

### 7.2 Experiment Logging

**MLflow Tracking:**
```python
with mlflow.start_run():
    mlflow.log_params({
        'formation_period': J,
        'holding_period': K,
        'skip_period': 5,
        'num_portfolios': 10
    })
    mlflow.log_metrics({
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'annual_return': cagr
    })
```

### 7.3 Documentation Standards

**Required Documentation:**
- Strategy logic flowchart
- Parameter sensitivity analysis
- Backtest assumptions and limitations
- Monthly performance attribution

---

## 8. Next Steps & Timeline

### Phase 1: Data Pipeline (Weeks 1-2)
- [ ] Load and validate OHLCV CSV files
- [ ] Verify adjusted closing prices are correct
- [ ] Create data quality monitoring for missing values
- [ ] Calculate daily returns from adjusted close prices

### Phase 2: Strategy Implementation (Weeks 3-4)
- [ ] Code baseline momentum strategy using price data only
- [ ] Implement portfolio construction with overlapping
- [ ] Add transaction cost modeling
- [ ] Create performance analytics module

### Phase 3: Backtesting & Analysis (Weeks 5-6)
- [ ] Run full historical backtest (2010-2025)
- [ ] Perform profit decomposition analysis
- [ ] Conduct robustness tests across subperiods
- [ ] Generate performance attribution reports

### Phase 4: Enhancement & Optimization (Weeks 7-8)
- [ ] Test signal enhancements (52-week high, volume-weighted momentum)
- [ ] Optimize J/K parameters with walk-forward analysis
- [ ] Implement risk management overlays using volatility
- [ ] Capacity and scalability analysis based on volume data

### Phase 5: Production Preparation (Weeks 9-10)
- [ ] Build real-time execution system
- [ ] Create monitoring dashboards
- [ ] Document trading rules and procedures
- [ ] Conduct paper trading validation

## Deliverables

1. **Research Report:** Comprehensive analysis following Jegadeesh & Titman methodology
2. **Backtest Results:** Full performance metrics using price data only
3. **Production Code:** Modular Python implementation for OHLCV data
4. **Execution Playbook:** Daily operational procedures for portfolio rebalancing
5. **Performance Dashboard:** Real-time tracking using price-based metrics

## Success Criteria

- **Minimum Performance:** IR > 0.5, Sharpe > 0.8
- **Consistency:** Positive returns in 60%+ of months
- **Robustness:** Strategy works across multiple subperiods
- **Scalability:** Can deploy 100M+ USD without significant impact
- **Risk Control:** Max drawdown < 25%, recovery < 12 months

---

## Appendix: Key Differences from Developed Markets

| Aspect | Developed Markets | Vietnam Market | Adaptation |
|--------|------------------|----------------|------------|
| Market Efficiency | High | Medium-Low | Stronger momentum expected |
| Shorting | Available | Prohibited | Long-only implementation |
| Liquidity | Deep | Limited | Stricter stock filtering |
| Transaction Costs | Low (5-10 bps) | Higher (25-50 bps) | Longer holding periods |
| Price Limits | Rare | Daily ±7% | Account for trading halts |
| Information Flow | Rapid | Slower | Extended momentum periods |

## Note on Simplifications

This research design focuses on the **pure price momentum strategy** as in the core sections (I-VII) of Jegadeesh & Titman (1993). We intentionally exclude:

1. **Earnings momentum analysis** - While the original paper examined returns around earnings announcements (Section VIII), this was exploratory analysis, not required for the strategy
2. **Fundamental data integration** - The beauty of momentum is its simplicity using only price data
3. **Sector/industry adjustments** - Not available with OHLCV data only
4. **Complex risk factors** - We use price-based volatility rather than fundamental risk measures

These simplifications align with the original paper's approach and allow for:
- **Exact replication** of the original methodology (which also used only price data)
- **Lower data costs** and reduced complexity
- **Focus on the core anomaly** without confounding factors
- **Cleaner academic comparison** to the original study

As Jegadeesh & Titman demonstrated, the momentum effect is robust using price data alone - no fundamental data required for the core strategy to work.