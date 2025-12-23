# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# %%

BULL = 0
BEAR = 1

regime_labels = {
    BULL: "Bull Market",
    BEAR: "Bear Market"
}


# %%
P = np.array([
    [0.90, 0.10],   
    [0.30, 0.70]    
])

# Sanity check
print("Row sums:", P.sum(axis=1))


# %%
T = 500 
regimes = np.zeros(T, dtype=int)
regimes[0] = BULL  

rng = np.random.default_rng()

for t in range(1, T):
    current = regimes[t - 1]
    regimes[t] = rng.choice([BULL, BEAR], p=P[current])


# %%
plt.figure(figsize=(14, 3))
plt.plot(regimes, drawstyle="steps-post")
plt.yticks([BULL, BEAR], ["Bull", "Bear"])
plt.xlabel("Time")
plt.ylabel("Market Regime")
plt.title("Simulated Market Regimes (Markov Chain)")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
BULL = 0
BEAR = 1


# %%
T = len(regimes)              
returns = np.zeros(T)

for t in range(T):
    state = regimes[t]        
    returns[t] = np.random.normal(
        mu[state],
        sigma[state]
    )


# %%
plt.figure(figsize=(14,4))
plt.plot(returns, linewidth=1)
plt.title("Regime-Switching Returns (Markov + Monte Carlo)")
plt.xlabel("Time")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Convert regimes to boolean masks
bull_mask = regimes == 0
bear_mask = regimes == 1

plt.figure(figsize=(14, 4))

# Plot Bull regime returns
plt.plot(
    np.where(bull_mask, returns, np.nan),
    color="green",
    linewidth=1,
    label="Bull Regime"
)

# Plot Bear regime returns
plt.plot(
    np.where(bear_mask, returns, np.nan),
    color="red",
    linewidth=1,
    label="Bear Regime"
)

plt.title("Regime-Switching Returns with Market States")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np

bull_returns = returns[regimes == 0]
bear_returns = returns[regimes == 1]

len(bull_returns), len(bear_returns)


# %%
def var_cvar(x, alpha=5):
    var = np.percentile(x, alpha)
    cvar = x[x <= var].mean()
    return var, cvar


# %%
var_bull, cvar_bull = var_cvar(bull_returns, 5)
var_bear, cvar_bear = var_cvar(bear_returns, 5)

var_bull, cvar_bull, var_bear, cvar_bear


# %%
import matplotlib.pyplot as plt

labels = ["Bull", "Bear"]
vars_ = [var_bull, var_bear]
cvars_ = [cvar_bull, cvar_bear]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, vars_, width, label="VaR (5%)")
plt.bar(x + width/2, cvars_, width, label="CVaR (5%)")

plt.xticks(x, labels)
plt.ylabel("Return")
plt.title("Regime-Specific Downside Risk")
plt.legend()
plt.grid(True)
plt.show()


# %%
mu_bull = bull_returns.mean()
sigma_bull = bull_returns.std()

mu_bear = bear_returns.mean()
sigma_bear = bear_returns.std()

mu_bull, sigma_bull, mu_bear, sigma_bear


# %%
T = 500  # horizon
simulated_returns = np.zeros(T)

for t in range(T):
    if regimes[t] == 0:  # Bull
        simulated_returns[t] = np.random.normal(mu_bull, sigma_bull)
    else:  # Bear
        simulated_returns[t] = np.random.normal(mu_bear, sigma_bear)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(14,4))
plt.plot(simulated_returns, linewidth=1)
plt.title("Regime-Switching Monte Carlo Simulation")
plt.xlabel("Time")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# %%
import numpy as np

# Separate simulated returns by regime
sim_bull = simulated_returns[regimes[:len(simulated_returns)] == 0]
sim_bear = simulated_returns[regimes[:len(simulated_returns)] == 1]

# 5% VaR
VaR_bull = np.percentile(sim_bull, 5)
VaR_bear = np.percentile(sim_bear, 5)

VaR_bull, VaR_bear


# %%
import matplotlib.pyplot as plt

N_PATHS = 200
T = len(regimes)

paths = np.zeros((N_PATHS, T))

for k in range(N_PATHS):
    for t in range(T):
        if regimes[t] == 0:
            paths[k, t] = np.random.normal(mu_bull, sigma_bull)
        else:
            paths[k, t] = np.random.normal(mu_bear, sigma_bear)

# Plot fan chart
plt.figure(figsize=(14,4))
for k in range(N_PATHS):
    plt.plot(paths[k], color="blue", alpha=0.05)

plt.title("Regime-Switching Monte Carlo Fan Chart")
plt.xlabel("Time")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# %%
def compute_drawdown(returns):
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return drawdown.min()

dd_bull = compute_drawdown(sim_bull)
dd_bear = compute_drawdown(sim_bear)

dd_bull, dd_bear


# %%

hidden_states = regimes.copy()


# %%
plt.figure(figsize=(14,4))

plt.plot(simulated_returns, color="gray", alpha=0.5)

plt.scatter(
    np.arange(len(simulated_returns)),
    simulated_returns,
    c=hidden_states,
    cmap="coolwarm",
    s=12
)

plt.title("Market Regimes (Markov Chain States)")
plt.xlabel("Time")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05

rolling_window = 50
var_series = np.full(len(returns), np.nan)

for t in range(rolling_window, len(returns)):
    window_returns = returns[t-rolling_window:t]
    var_series[t] = np.percentile(window_returns, alpha * 100)

plt.figure(figsize=(14,4))
plt.plot(var_series, color="black", linewidth=1.2, label="Rolling VaR (5%)")
plt.plot(returns, alpha=0.3, label="Returns")
plt.title("Regime-Conditioned Rolling VaR")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.show()


# %%
n_paths = 300
T = 250

paths = np.zeros((n_paths, T))

for i in range(n_paths):
    state = regimes[0]
    for t in range(T):
        if state == BULL:
            ret = np.random.normal(mu_bull, sigma_bull)
        else:
            ret = np.random.normal(mu_bear, sigma_bear)

        paths[i, t] = ret
        state = np.random.choice([BULL, BEAR], p=P[state])

cumulative_paths = np.cumsum(paths, axis=1)

plt.figure(figsize=(14,5))
plt.plot(cumulative_paths.T, color="gray", alpha=0.05)
plt.title("Regime-Switching Monte Carlo Fan Chart")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()


# %%
def max_drawdown(series):
    cumulative = np.cumsum(series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return np.min(drawdown)

bull_dd = max_drawdown(returns[regimes == BULL])
bear_dd = max_drawdown(returns[regimes == BEAR])

plt.figure(figsize=(6,4))
plt.bar(["Bull Regime", "Bear Regime"], [bull_dd, bear_dd],
        color=["green", "red"])
plt.title("Expected Maximum Drawdown by Regime")
plt.ylabel("Drawdown")
plt.grid(True)
plt.show()



