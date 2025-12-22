# Monte Carlo Portfolio Risk Engine

## Overview
This project implements a **Monte Carlo–based portfolio risk engine** from scratch using Python.  
The objective is to simulate realistic future portfolio return distributions and quantify **downside risk** using modern risk measures such as **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)**.

The engine compares:
- Classical **Gaussian (Normal)** assumptions  
- **Fat-tailed (Student-t)** risk models  

to highlight how traditional models underestimate extreme losses.

---

## Why this project matters
Financial returns are **not perfectly normal**.  
Extreme events occur more often than Gaussian models predict.

This project demonstrates:
- How correlation structures drive portfolio risk
- Why fat tails matter for downside risk
- How Monte Carlo simulation samples *possible futures*, not point forecasts

The goal is **risk understanding**, not prediction.

---

## Core concepts used
- Empirical mean and covariance estimation  
- Cholesky decomposition for correlation modeling  
- Multivariate Monte Carlo simulation  
- Value at Risk (VaR)  
- Conditional Value at Risk (CVaR)  
- Tail risk comparison (Gaussian vs Student-t)
