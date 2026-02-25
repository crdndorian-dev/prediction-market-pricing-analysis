Author: Dorian Cardon
Institution: Université Paris II Panthéon-Assas  
Year: 2026  
Contact: crdn.dorian@gmail.com
GitHub: https://github.com/crdndorian-dev

# Abstract

This project develops a quantitative research framework designed to analyze the pricing structure and potential inefficiencies of Polymarket-style binary prediction markets. The objective is not to claim superior forecasting accuracy, but rather to explore whether prices observed in decentralized prediction markets are consistent with economically grounded benchmarks derived from traditional financial markets.

Polymarket contracts resemble digital options: they deliver a fixed payoff conditional on the realization of a binary event at a specified expiration date. Their prices can therefore be interpreted as implied probabilities. However, unlike exchange-listed derivatives, Polymarket prices are formed in environments characterized by heterogeneous beliefs, varying liquidity, retail participation, and limited arbitrage enforcement. As a result, quoted probabilities may reflect sentiment, positioning effects, or microstructure noise in addition to information about expected outcomes.

In contrast, listed equity options markets provide a rich cross-section of arbitrage-constrained prices. Under standard no-arbitrage assumptions, these prices encode risk-neutral probabilities that can be extracted using option pricing theory. While risk-neutral probabilities do not correspond directly to real-world probabilities due to risk premia and demand imbalances, they provide a coherent and economically disciplined reference point.

The central contribution of this project is to construct a systematic and time-safe pipeline that links these two environments. The framework first extracts risk-neutral probabilities from listed options data for events comparable to those traded on Polymarket. It then investigates, through statistical calibration methods, whether a stable relationship can be learned between risk-neutral measures and realized event frequencies. This mapping is subsequently used to generate fair-value probability estimates for Polymarket contracts, allowing for a structured comparison between decentralized market prices and derivative-implied benchmarks.

The project places particular emphasis on dataset design, temporal consistency, and robustness across different market regimes. Rather than assuming persistent inefficiency, it evaluates empirically whether deviations between Polymarket prices and calibrated benchmarks are economically meaningful, statistically significant, or largely attributable to noise. Backtesting procedures are implemented to assess whether observed discrepancies translate into consistent expected value under realistic assumptions.

# How to use

To launch the webapp locally, run:

```bash
./run-webapp.sh
```

This script provisions the backend venv/deps, starts FastAPI on port 8000 (or the next free port in 8000-8050), and starts the Vite frontend wired to that API. Use the URL printed by Vite in the terminal. For manual setup, see `src/webapp/README.md`.
