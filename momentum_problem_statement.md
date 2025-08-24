## Problem Statement: Momentum-Based Trading Strategy Research

### 1. Objective / Research Question

- **Core Strategy Framework:** Investigate momentum-based relative strength trading strategies adapted for Vietnam market constraints. Given no short selling allowed, the strategy involves:
    
    - **Winner Portfolio (Long):** Buy stocks with highest past returns (top quintile/decile)
        
    - **Benchmark Comparison:** Compare against market index or equal-weighted universe
        
    - **Alternative Approaches:** Long-only momentum vs. momentum-tilt portfolios
        
    - **Risk-Adjusted Performance:** Focus on alpha generation rather than zero-cost returns
        

- **Key Strategic Decisions:**

	- **Formation Period (J):** 3, 6, 9, or 12 months for ranking past performance
	    
	- **Holding Period (K):** 1, 3, 6, or 12 months for maintaining positions
	    
	- **Skip Period:** 1 week between formation and holding to avoid microstructure effects
	    
	- **Portfolio Construction:** Equal-weighted decile portfolios
	    
	- **Rebalancing:** Monthly with overlapping portfolios for continuous exposure
    

---

### 2. Market & Asset Scope

- Focus on Vietnam equity markets (HOSE & HSX listed stocks)
- Sample period: January 2010 to August 2025 (15+ years)
- All stocks with sufficient return history and trading liquidity

---

### 3. Data Inputs & Sources

- **Stock Price and Return Data:**
    
    - Daily OHLCV data from Vietnam stock exchanges (HOSE & HSX)
        
    - Adjusted closing prices for stock splits and dividends
        
    - Trading volume for liquidity filtering
        
    - All listed stocks with minimum trading history
        
- **Return Calculations:**
    
    - Daily returns: `(Close_t / Close_{t-1}) - 1`
        
    - Monthly returns: Compounded daily returns per month
        
    - Formation period returns: J-month cumulative returns
        
    - Skip period: 1 week gap between formation and holding periods

---

### 4. Hypothesis / Signal Assumption

- **Prediction 1: Return Continuation (Momentum in Vietnam)**  
    _Hypothesis_: Stocks with high past returns will continue outperforming in subsequent periods, even in emerging market context  
    _Test_: Statistical significance of winner portfolio excess returns over benchmark
    
- **Prediction 2: Optimal Horizon Effects**  
    _Hypothesis_: Momentum is strongest at intermediate horizons (3-6 months) in Vietnam market  
    _Test_: Compare excess returns across different J/K combinations
    
- **Prediction 3: Information Processing in Emerging Markets**  
    _Hypothesis_: Momentum effects may be stronger in Vietnam due to less efficient information processing  
    _Test_: Compare momentum strength with developed market benchmarks
    
- **Prediction 4: Long-only Strategy Effectiveness**  
    _Hypothesis_: Long-only momentum strategy can generate positive alpha despite inability to short losers  
    _Test_: Track portfolio performance vs. VN-Index over multiple market cycles

---

### 5. Modeling Approach

#### Signal Generation

**Momentum Score Calculation:**

```python
# J-month return ending 1 week before holding period
momentum_score = (price_t / price_{t-J*21}) - 1
skip_period = 5  # 1 week skip period
formation_end = t - skip_period
```

**Ranking Process:**
1. Calculate J-month returns for all eligible stocks
2. Rank stocks in ascending order of past returns
3. Form 10 equal-weighted decile portfolios
4. Decile 1 = "Losers" (lowest returns), Decile 10 = "Winners" (highest returns)

#### Portfolio Construction

**Long-Only Momentum Strategy (Adapted for Vietnam):**
- **Primary Approach:** Long position in top quintile/decile (Winners only)
- **Benchmark:** Compare against VN-Index or equal-weighted market portfolio
- **Alternative:** Momentum-tilt portfolio with overweight winners, underweight losers

**Position Sizing Options:**
- Equal-weighted winner portfolio
- Market-cap weighted with momentum tilt
- Risk-budgeted allocation based on volatility

**Overlapping Portfolio Method:**
- Maintain K overlapping portfolios simultaneously
- Each month: close 1/K old positions, open 1/K new positions
- Provides continuous exposure while controlling turnover

#### Statistical Testing Framework

**One-Factor Model Analysis:**
```
r_it = μ_i + b_i*f_t + e_it

Where:
- r_it = return on security i at time t  
- f_t = factor return (market)
- e_it = firm-specific return component
```

**Profit Decomposition:**
Expected momentum profits decomposed into:
1. Cross-sectional dispersion in expected returns
2. Serial correlation in factor returns  
3. Serial correlation in firm-specific returns

---

### 6. Constraints & Risks

- **Market-Specific Constraints:**
    - **No Short Selling:** Vietnam stock market prohibits short selling, requiring long-only strategies
    - **Limited Liquidity:** Smaller market with potentially lower trading volumes
    - **Foreign Ownership Limits:** Foreign investors face ownership restrictions on certain stocks
    - **Settlement and Trading Rules:** T+2 settlement and daily price limits (±7% for most stocks)

- **Implementation Constraints:**
    - Transaction costs may be higher than developed markets
    - Portfolio rebalancing frequency limited by liquidity constraints
    - Market concentration in few large-cap stocks

- **Risk Considerations:**
    - Emerging market volatility and systematic risk
    - Currency risk for foreign investors
    - Regulatory changes affecting market access or trading rules
    - Strategy cannot benefit from short-side alpha due to shorting restrictions

---

### 7. Evaluation Criteria

- **Primary Metrics:** 
    - Excess returns over VN-Index and equal-weighted benchmarks
    - Information Ratio (excess return / tracking error)
    - Sharpe Ratio and risk-adjusted performance metrics
    
- **Risk Assessment:** 
    - Portfolio beta relative to market index
    - Maximum drawdown and volatility analysis
    - Analysis across market-cap segments
    
- **Robustness Tests:**
    - Performance across different market cycles (2010-2025)
    - Seasonal patterns specific to Vietnam market
    - Transaction cost impact analysis for realistic implementation

---

### 8. Expected Findings

#### Core Results (6-month/6-month strategy adapted for Vietnam):
- **Excess Returns:** Target 2-5% annual excess return over VN-Index (lower than original study due to long-only constraint)
- **Information Ratio:** Aim for IR > 0.5 indicating consistent outperformance
- **Statistical Significance:** t-statistics >2.0 for momentum strategy excess returns
- **Optimal Strategy:** Expected to be 3-6 month formation, 3-6 month holding periods

#### Vietnam Market Specific Expectations:
- **Emerging Market Premium:** Higher volatility but potentially stronger momentum effects
- **Limited Short-Side Benefits:** Performance will be lower than zero-cost studies due to inability to short losers  
- **Concentration Risk:** Higher impact from few large stocks on momentum strategy performance
- **Liquidity Constraints:** May require longer holding periods than optimal

#### Market Efficiency Implications:
- **Information Processing:** Expected delayed reactions to information in emerging market context
- **Momentum Duration:** May differ from developed markets due to different investor sophistication
- **Transaction Costs:** Higher costs may reduce net profitability compared to theoretical returns

---

### 9. Signal Validation Framework

- **In-Sample Analysis:**
    - Full sample results (2010-2025) for strategy optimization
    - Robustness across market-cap based subsamples (large, mid, small cap)
        
- **Out-of-Sample Testing:**
    - Rolling window validation with 3-year training, 1-year testing periods
    - Walk-forward analysis to test parameter stability
    
- **Vietnam Market Context:**
    - Performance during major market events (2011 crisis, 2018 correction, COVID-19 impact)
    - Comparison with traditional value and growth strategies in Vietnam
    - Analysis of momentum persistence across different economic cycles
    
- **Risk Controls:**
    - Benchmark-adjusted returns using VN-Index as market proxy
    - Analysis of sector concentration and style bias
    - Liquidity-adjusted performance metrics

#### Performance Benchmarks:
- **Success Criteria:** Consistent positive excess returns over rolling 3-year periods
- **Risk Threshold:** Maximum drawdown < 25%, beta between 0.8-1.2
- **Persistence Test:** Outperformance in at least 60% of months and positive annual returns in 70% of years