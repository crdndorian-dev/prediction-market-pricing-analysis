# Real-World Probability Calibration for Polymarket Contracts
### An Options-Implied, Time-Safe Probabilistic Modeling Pipeline

## Abstract

This project implements a **probability calibration framework** designed to estimate *real-world event probabilities* for Polymarket-style binary contracts using information extracted from listed options markets.

Polymarket contracts behave like **digital options** (binary payoff, fixed expiry), but their market prices embed heterogeneous beliefs, liquidity effects, and microstructure noise.  
In contrast, listed options markets provide a dense cross-section of prices that encode **risk-neutral probabilities (pRN)** under no-arbitrage assumptions.

The core objective of this project is to **learn a stable and time-safe mapping from risk-neutral probabilities to real-world probabilities**, and to apply this mapping to Polymarket contracts in order to identify statistically meaningful discrepancies (edges).

---

## Local Web App (UI)

Run the backend and frontend in separate terminals:

Backend:
```bash
cd src/webapp/backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000
```

Frontend:
```bash
cd src/webapp/frontend
npm install
npm run dev
```

---

## Conceptual Framework

Let:

- \( p_{RN} \): risk-neutral probability implied by options prices  
- \( p_{RW} \): real-world probability of the event (unobservable)  
- \( p_{PM} \): Polymarket-implied probability (market price)

Options markets price under the **risk-neutral measure**, while Polymarket contracts settle under the **physical (real-world) measure**.

This project models:
\[
p_{RW} = f(p_{RN}, X)
\]
where:
- \( f \) is a calibrated probabilistic mapping
- \( X \) is a vector of auxiliary features (time-to-expiry, moneyness, volatility proxies, etc.)

The calibrated estimate \( \hat{p}_{RW} \) (denoted **pHAT**) is then compared to \( p_{PM} \) to assess potential edge.

---

## Mathematical Intuition

### Risk-Neutral vs Real-World Probabilities

Options markets price payoffs under the **risk-neutral measure** \( \mathbb{Q} \), not under the real-world (physical) measure \( \mathbb{P} \).

For a digital-style payoff:
\[
\mathbb{1}\{ S_T > K \}
\]

The option-implied probability extracted from prices corresponds to:
\[
p_{RN} = \mathbb{Q}(S_T > K)
\]

However, the quantity of interest for Polymarket settlement is:
\[
p_{RW} = \mathbb{P}(S_T > K)
\]

In general:
\[
\mathbb{Q}(S_T > K) \neq \mathbb{P}(S_T > K)
\]

The difference arises due to:
- Risk premia
- Volatility risk
- Skew and tail pricing
- Market participants’ hedging demand
- Structural and behavioral effects

---

### Why a Logistic Calibration?

We model the relationship between \( p_{RN} \) and \( p_{RW} \) using a **logistic link** for several reasons:

1. **Probability constraints**  
   The logistic function naturally enforces:
   \[
   \hat{p}_{RW} \in (0, 1)
   \]

2. **Monotonicity**  
   Empirically, higher risk-neutral probabilities should correspond to higher real-world probabilities, though not linearly.

3. **Interpretability**  
   The model provides an explicit parametric mapping between the two measures.

---

### Logit Space as the Natural Domain

Rather than modeling probabilities directly, the calibration is performed in **log-odds space**:

\[
\text{logit}(p) = \log\left( \frac{p}{1 - p} \right)
\]

This choice is motivated by:
- Approximate symmetry in log-odds space
- Stability near the boundaries \( 0 \) and \( 1 \)
- Linearization of multiplicative distortions between measures

The core calibration equation is:

\[
\text{logit}(p_{RW}) = \alpha + \beta \cdot \text{logit}(p_{RN}) + \gamma^\top X
\]

where:
- \( \alpha \) is a global intercept
- \( \beta \) captures systematic distortion between measures
- \( X \) are auxiliary conditioning features

---

### Interpretation of Parameters

- \( \beta < 1 \): risk-neutral probabilities are overconfident (too extreme)
- \( \beta > 1 \): risk-neutral probabilities are underconfident
- \( \alpha \neq 0 \): global directional bias
- \( \gamma \): conditional adjustments (e.g., time-to-expiry, moneyness effects)

The calibrated real-world probability estimate is then:

\[
\hat{p}_{RW} = \sigma\left( \alpha + \beta \cdot \text{logit}(p_{RN}) + \gamma^\top X \right)
\]

where \( \sigma(\cdot) \) is the logistic function.

---

### Loss Functions as Proper Scoring Rules

The model is trained using **proper scoring rules**, ensuring honest probability estimation:

- **Log Loss**:
\[
\mathcal{L}_{\text{log}} = - \left[ y \log(\hat{p}) + (1 - y)\log(1 - \hat{p}) \right]
\]

- **Brier Score**:
\[
\mathcal{L}_{\text{brier}} = (y - \hat{p})^2
\]

- **Expected Calibration Error (ECE)**:
\[
\text{ECE} = \sum_{b} | \mathbb{E}[y \mid \hat{p} \in b] - \mathbb{E}[\hat{p} \mid \hat{p} \in b] |
\]

These metrics emphasize **calibration quality**, not directional accuracy.

---

### Why This Matters for Edge Detection

Polymarket prices approximate:
\[
p_{PM} \approx \mathbb{E}_{\text{participants}}[\mathbb{P}(S_T > K)]
\]

The calibrated estimate \( \hat{p}_{RW} \) provides an alternative, model-based estimate of the same quantity.

An edge exists only if:
\[
\hat{p}_{RW} - p_{PM}
\]
is sufficiently large **after accounting for uncertainty, spreads, and execution constraints**.

This framework does not assume persistent inefficiency, but enables **systematic, probabilistic comparison** across markets.


## Pipeline Overview

The pipeline is deliberately modular and time-safe, and consists of four main stages.

---

## 1. Historical Options Snapshot Construction

A dedicated data-ingestion script builds a **historical panel of option snapshots**.

This stage:
- Collects historical options chains (primarily calls)
- Aligns each snapshot with a specific strike and expiry
- Computes relevant structural features
- Extracts **risk-neutral probabilities (pRN)** using the option surface
- Produces a clean, point-in-time CSV dataset

Key design principles:
- No future information leakage
- Each row represents an information set available *at the time*
- The dataset is suitable for strict out-of-sample validation

This dataset serves as the **training universe** for probability calibration.

---

## 2. Probability Calibration Model (pRN → pHAT)

The calibration step estimates the mapping from risk-neutral probabilities to real-world probabilities.

### Model Class

- **Logistic regression** (parametric, interpretable, robust)
- Primary regressor: `logit(pRN)`
- Optional auxiliary features for conditional calibration
- Optional per-ticker intercept adjustments

### Estimation

- Parameters are estimated via **maximum likelihood**
- Regularization is used to avoid overfitting
- Training is **strictly time-ordered**

### Validation & Model Selection

The model is evaluated using rolling, contiguous validation windows.

Primary loss functions:
- **Log Loss** (proper scoring rule)
- **Brier Score**
- **Expected Calibration Error (ECE)**

The emphasis is not on classification accuracy, but on **probability quality and calibration stability**.

The output of this stage is a calibrated function:
\[
\hat{p}_{RW} = \text{logistic}(\alpha + \beta \cdot \text{logit}(p_{RN}) + \gamma^\top X)
\]

---

## 3. Polymarket Market Snapshot Ingestion

A separate script ingests **live or historical Polymarket contracts**.

For each contract, it:
- Extracts market prices and spreads
- Identifies the underlying real-world event
- Matches the contract to a **structurally equivalent listed option**
  - Same underlying
  - Same strike
  - Same expiry
- Computes the corresponding **pRN** using options data

The result is a Polymarket snapshot enriched with option-implied probabilities.

---

## 4. Real-World Probability Estimation & Edge Analysis

In the final step, the calibrated model is applied to Polymarket snapshots.

This stage:
- Transforms pRN → pHAT using the trained calibration model
- Computes both YES and NO real-world probabilities
- Compares pHAT to Polymarket market prices
- Produces edge diagnostics suitable for downstream execution logic

This step intentionally **does not perform trade execution**.  
It outputs probabilistic assessments only.

---

## Design Philosophy

- **Calibration over prediction**  
  The goal is not to forecast prices, but to estimate probabilities accurately.

- **Time safety is non-negotiable**  
  All training, validation, and evaluation respect temporal ordering.

- **Model simplicity is intentional**  
  Logistic calibration is chosen for stability, interpretability, and robustness.

- **Market realism**  
  Risk-neutral probabilities are treated as informative but biased estimators of real-world probabilities.

---

## Limitations & Assumptions

- Risk-neutral probabilities are extracted assuming sufficient option liquidity
- Calibration stability may degrade in regime shifts
- Polymarket prices may be affected by microstructure noise and participation bias
- No claim of persistent edge is made

---

## Intended Use

This project is intended as:
- A **research framework** for probabilistic calibration
- A **signal-generation layer**, not a full trading system
- A structured input for decision-making, risk management, or further modeling

---

## Disclaimer

This repository is for research and educational purposes only.  
Financial markets are adversarial and non-stationary.  
No guarantees of profitability are implied.

---

## Author

Built with quantitative rigor, skepticism, and curiosity.
