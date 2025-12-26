# Stochastic Realities: From Monte Carlo Worlds to Model Risk

This project is a deep, end-to-end exploration of **financial randomness, pricing models, and model risk**, built progressively from first principles to advanced stochastic volatility frameworks.

Rather than treating Monte Carlo, Markov chains, or Black–Scholes as isolated tools, this project **connects them into a single evolving narrative**:  
how assumptions shape markets, how models diverge from reality, and why hedging errors are unavoidable when volatility itself is random.

The project is structured into **four conceptual modules**, each with its own mathematical, computational, and financial focus.

---
## Project Structure

```text
MonteCarlo_Portfolio_Project/
├── Scripts/
│   ├── Module1_MonteCarlo_from_scratch.py
│   ├── Module2_Regime_Switching_MonteCarlo.py
│   ├── Module3_Black_Scholes_Greeks&Options.py
│   └── Module4_Model_Risk_Hedging_Errors.py
│
├── Outputs/
│   ├── Figures_Monte_Carlo/
│   ├── Figures_Markov_Chains/
│   ├── Figures_Black_Scholes/
│   └── Figures_Beyond_Black_Scholes/




## Module 1 — Monte Carlo From First Principles

**Core idea:**  
Monte Carlo is not a pricing trick — it is a *probability engine*.

### What this module does
- Builds Monte Carlo simulations **from scratch**
- Simulates asset price paths using geometric Brownian motion
- Visualizes how randomness aggregates into distributions
- Interprets Monte Carlo paths as **ensembles of possible financial worlds**

### Key insights
- Expectation emerges from chaos
- Distributional thinking is more important than point forecasts
- Pricing is secondary — uncertainty is primary

Monte Carlo is introduced here as the **foundation of everything that follows**, not as an add-on.

---

## Module 2 — Regime Switching & Markov Chains (The Hidden State)

**Core idea:**  
Markets do not live in one volatility regime.

### What this module does
- Introduces **Markov chains** to model regime changes
- Simulates switching between low- and high-volatility states
- Couples regime states with Monte Carlo price evolution
- Demonstrates **path-dependence driven by hidden states**

### Key insights
- Volatility clustering is structural, not noise
- Identical shocks behave differently across regimes
- Memory enters markets through state persistence

Here, **Markov chains become the hero**, explaining why static assumptions fail.

---

## Module 3 — Black–Scholes, Greeks, and the Illusion of Stability

**Core idea:**  
Black–Scholes works beautifully — under assumptions that rarely hold.

### What this module does
- Implements Black–Scholes option pricing
- Computes option Greeks (Delta, Gamma, Vega)
- Prices options using Monte Carlo vs closed-form solutions
- Examines sensitivity to volatility and strike

### Key insights
- Flat volatility implies a flat implied volatility surface
- Greeks assume a world where volatility is known and constant
- Pricing accuracy does not imply hedging accuracy

This module shows **why Black–Scholes is internally consistent — yet externally fragile**.

---

## Module 4 — Beyond Black–Scholes: Stochastic Volatility & Model Risk

**Core idea:**  
When volatility moves randomly, hedging breaks.

### What this module does
- Introduces **stochastic volatility dynamics (Heston-style intuition)**
- Visualizes leverage effects (negative correlation between price and volatility)
- Compares terminal price distributions: Black–Scholes vs stochastic volatility
- Analyzes **delta-hedging errors under model mismatch**

### Key insights
- Identical options → radically different tail risks
- Volatility smiles emerge naturally from stochastic volatility
- Hedging errors accumulate even with continuous rebalancing
- Model risk is not a bug — it is structural

This module concludes the project by exposing the **limits of classical risk-neutral thinking**.

---

## Final Takeaway

This project demonstrates that:

- Monte Carlo is a language, not a technique
- Regime switching explains market memory
- Black–Scholes is elegant but incomplete
- Stochastic volatility turns pricing risk into **model risk**
- Hedging errors are mathematically inevitable under realism

The journey moves from **points → paths → states → distributions → tail risk**, forming a coherent geometric and probabilistic understanding of markets.

---

## Technologies & Concepts Used

- Python (NumPy, SciPy, Matplotlib)
- Monte Carlo simulation
- Markov chains & regime switching
- Black–Scholes pricing
- Greeks & sensitivity analysis
- Stochastic volatility intuition
- Hedging error analysis
- Distributional and tail-risk thinking




