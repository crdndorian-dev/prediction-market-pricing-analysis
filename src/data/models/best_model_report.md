# Best Auto-Selected Model

- **objective** (rolling VAL logloss mean): 0.5410417120153275
- **validation logloss std across windows**: 0.05112175771795304
- **test metrics (not used for selection)**: logloss=0.524470681765587, brier=0.17311522459769033, ece=0.06127926091722606

## Final configuration

- C: 0.5
- Features: ['x_logit_prn', 'log_m_fwd', 'abs_log_m_fwd']
- Ticker intercepts: non_foundation
- Ticker interactions: False
- Recency decay half-life (weeks): 0.0
- Calibration (Platt): none
- Foundation enabled: False
- Foundation tickers: 
- Foundation weight: 1.0

## Parameter family choices

1. C: 0.5
2. features: ['x_logit_prn', 'log_m_fwd', 'abs_log_m_fwd']
3. ticker_intercepts: non_foundation
4. ticker_interactions: False
5. recency_decay_weeks: 0.0
6. calibration: none
7. foundation_enabled: False
8. foundation_weight: 1.0

## Trial provenance
- trial id: 24
- trial dir: /Users/dorianc./Desktop/polyedgetool/prediction-market-pricing-analysis/src/data/models/trials/024__C=0.5__feat=moneyness__ti=non_foundation__tx=0__decay=0__platt=none__foundation=off__2a8f138e
- top coefficients summary: not captured (run calibrator manually for coefficient inspection)