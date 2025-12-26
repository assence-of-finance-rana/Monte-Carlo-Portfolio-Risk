# Stochastic Realities: From Monte Carlo Worlds to Model Risk

This project is a **deep, end-to-end exploration of financial randomness**, asset pricing, and **model risk**, built progressively from first principles to advanced stochastic volatility frameworks.

Rather than treating Monte Carlo simulation, Markov chains, or Black–Scholes as isolated tools, this project **connects them into a single evolving narrative**:
how uncertainty propagates through markets, how assumptions shape prices, and why **model risk is unavoidable** when volatility itself is random.

The project is structured into **four conceptual modules**, each with a clear mathematical, computational, and financial focus.

---

## Project Structure

```text
MonteCarlo_Portfolio_Project/
│
├── Scripts/
│   ├── Module1_MonteCarlo_from_scratch.py
│   ├── Module2_Regime_Switching_MonteCarlo.py
│   ├── Module3_Black_Scholes_Greeks&Options.py
│   └── Module4_Model_Risk_Hedging_Errors.py
│
└── Outputs/
    ├── Figures_Monte_Carlo/
    ├── Figures_Markov_Chains/
    ├── Figures_Black_Scholes/
    └── Figures_Beyond_Black_Scholes/
```

---

## Module 1 — Monte Carlo From First Principles

**Core idea:**  
Monte Carlo is not a pricing trick — it is a **probability engine**.

### What this module does
- Builds Monte Carlo simulations **from scratch**
- Simulates asset price paths using **Geometric Brownian Motion**
- Visualizes how randomness aggregates into distributions
- Interprets Monte Carlo paths as **ensembles of possible financial worlds**

### Key insights
- Expectation emerges from chaos  
- Distributional thinking matters more than point forecasts  
- Pricing is secondary — **uncertainty is primary**

Monte Carlo is introduced here as the **foundation of everything that follows**, not as an add-on.

---

## Module 2 — Regime Switching & Markov Chains (The Hidden State)

**Core idea:**  
Markets do not live in a single volatility regime.

### What this module does
- Introduces **Markov chains** to model regime changes
- Simulates switching between low- and high-volatility states
- Couples regime states with Monte Carlo price evolution
- Demonstrates **path dependence driven by hidden states**

### Key insights
- Volatility clustering is structural, not noise  
- Identical shocks behave differently across regimes  
- Memory enters markets through **state persistence**

This module explains *why* simple Monte Carlo assumptions break down in real markets.

---

## Module 3 — Black–Scholes, Greeks & Risk-Neutral Pricing

**Core idea:**  
Black–Scholes works because of assumptions — and fails because of them.

### What this module does
- Derives option pricing under **risk-neutral valuation**
- Implements Black–Scholes pricing from first principles
- Computes Greeks (Delta, Gamma, Vega)
- Visualizes payoff structures and sensitivity profiles

### Key insights
- Risk-neutral pricing is a **measure change**, not a belief  
- Greeks describe **local** risk, not global uncertainty  
- Constant volatility is a mathematical convenience, not reality

This module formalizes pricing — and quietly exposes its fragility.

---

## Module 4 — Beyond Black–Scholes: Stochastic Volatility & Model Risk

**Core idea:**  
Hedging fails when volatility itself is random.

### What this module does
- Introduces **stochastic volatility** (Heston-style dynamics)
- Simulates leverage effect (negative price–volatility correlation)
- Compares Black–Scholes vs stochastic volatility outcomes
- Quantifies **hedging errors** under model misspecification
- Analyzes tail risk, skew, and convexity effects

### Key insights
- Model risk is structural, not accidental  
- Delta hedging fails under stochastic volatility  
- Tail behavior dominates real-world risk  
- Different models imply **different financial realities**

This module closes the loop: from pricing models to **decision-making under uncertainty**.

---

## Overall Takeaway

This project is not about implementing models —  
it is about **understanding what models assume, what they ignore, and how they fail**.

From Monte Carlo worlds to regime shifts, from Black–Scholes elegance to stochastic volatility chaos, the project builds a **coherent mental framework** for thinking about risk in real financial systems.

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- Plotly (for interactive visualizations)
- Monte Carlo simulation
- Markov chains
- Black–Scholes framework
- Greeks
- Stochastic volatility
- Model risk analysis

---

## Intended Audience
- Finance students exploring quantitative methods
- Aspiring risk analysts and derivatives researchers
- Anyone interested in **how randomness becomes risk**


