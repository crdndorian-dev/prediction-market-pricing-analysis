# Best Auto-Selected Model

- **score** (composite with complexity penalty): 0.4805021153431608
- **objective** (rolling VAL logloss mean): 0.4805021153431608
- **delta_brier**: N/A
- **delta_ece**: N/A
- **validation logloss std across windows**: 0.024473944456228598
- **test metrics (not used for selection)**: logloss=0.4021984837847738, brier=0.12550581087997198, ece=0.05556615191837773

## Final configuration

- C: 0.05
- Features: ['x_logit_prn']
- Feature choices: {'volatility': [], 'moneyness': []}
- Interactions: x_m=False, x_abs_m=False
- Ticker intercepts: none
- Ticker interactions: False
- Recency decay half-life (weeks): 26.0
- Calibration: none
- Group reweight: none
- Foundation enabled: False
- Foundation tickers: 
- Foundation weight: 1.0

## Parameter family choices

1. C: 0.05
2. features: ['x_logit_prn']
3. feature_choices: {'volatility': [], 'moneyness': []}
4. enable_x_m: False
5. enable_x_abs_m: False
6. ticker_intercepts: none
7. ticker_interactions: False
8. recency_decay_weeks: 26.0
9. calibration: none
10. group_reweight: none
11. foundation_enabled: False
12. foundation_weight: 1.0

## Trial provenance
- trial id: 52
- trial dir: /Users/dorianc./Desktop/polyedgetool/polyedgetool/src/data/models/1dte-logit-model/trials/052__C=0.05__feat=min__ti=none__tx=0__decay=26__cal=none__foundation=off__8635e52e
- top coefficients summary: not captured (run calibrator manually for coefficient inspection)