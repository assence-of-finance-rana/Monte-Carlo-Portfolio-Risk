# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Kernel live ")


# %%
# Load datasets
aapl_raw = pd.read_csv("data/AAPL_MONTHLY.csv")
portfolio_raw = pd.read_excel("data/VBTLX_VFIAX.xlsx")

aapl_raw.head(), portfolio_raw.head()


# %%
# Clean AAPL data
aapl = aapl_raw[["Date", "Adj Close"]].copy()
aapl["Date"] = pd.to_datetime(aapl["Date"])

# Monthly returns
aapl["AAPL"] = aapl["Adj Close"].pct_change()
aapl_returns = aapl[["AAPL"]].dropna()

aapl_returns.head()


# %%
# Extract portfolio returns by position
portfolio = portfolio_raw.iloc[:, 1:3].copy()
portfolio.columns = ["VBTLX", "VFIAX"]

portfolio = portfolio.dropna()
portfolio.head()


# %%
returns = pd.concat(
    [portfolio, aapl_returns],
    axis=1
).dropna()

returns.head()


# %%
returns_mc = returns.copy()

# Convert mutual fund % returns to decimals
returns_mc[["VBTLX", "VFIAX"]] = returns_mc[["VBTLX", "VFIAX"]] / 100

returns_mc.head()


# %%
mu = returns_mc.mean()
mu


# %%
cov = returns_mc.cov()
cov


# %%
returns_mc = returns_mc.astype(float)

mu = returns_mc.mean()
cov = returns_mc.cov()


# %%
import numpy as np

L = np.linalg.cholesky(cov)
L


# %%
T = 120
N = len(mu)

Z = np.random.normal(0, 1, size=(T, N))


# %%
L = np.linalg.cholesky(cov)
L


# %%
L @ L.T


# %%
correlated_shocks = Z @ L.T
correlated_shocks.shape


# %%
pd.DataFrame(
    correlated_shocks,
    columns=returns_mc.columns
).head()


# %%
np.cov(correlated_shocks.T)


# %%
n_sims = 10000
n_assets = 3
(n_sims, n_assets)
drift = mu.values
returns_simulated = correlated_shocks + drift


# %%
returns_simulated.shape



# %%
L = cholesky(cov)

L

# %%
Z = np.random.normal(size=(n_sims, n_assets))
correlated_shocks = Z @ L.T
correlated_shocks.shape


# %%
weights = np.array([0.6785, 0.3215, 0.0])  
weights.sum()


# %%
n_sims = 10000
n_assets = returns_mc.shape[1]

Z = np.random.normal(size=(n_sims, n_assets))
L = np.linalg.cholesky(cov)

correlated_shocks_mc = Z @ L.T
correlated_shocks_mc.shape


# %%
drift = mu.values
returns_simulated_mc = correlated_shocks_mc + drift
returns_simulated_mc.shape


# %%
portfolio_returns_mc = returns_simulated_mc @ weights
portfolio_returns_mc.shape


# %%
portfolio_returns.mean(), portfolio_returns.std()


# %%
import matplotlib.pyplot as plt

plt.hist(portfolio_returns_mc, bins=100, density=True)
plt.title("Monte Carlo Portfolio Return Distribution")
plt.xlabel("Return")
plt.ylabel("Density")
plt.show()


# %%
plt.savefig("outputs/figures/monte_carlo_distribution.png", dpi=300, bbox_inches="tight")


# %%
portfolio_returns.shape
portfolio_returns.mean()
portfolio_returns.std()


# %%
var_95 = np.percentile(portfolio_returns, 5)
var_99 = np.percentile(portfolio_returns, 1)

var_95, var_99


# %%
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()

cvar_95, cvar_99


# %%
np.mean(portfolio_returns <= var_95)



# %%
from scipy.stats import kurtosis, skew

skew(portfolio_returns), kurtosis(portfolio_returns)



# %%
from scipy.stats import t

df = 5  # degrees of freedom (lower = fatter tails)

z_t = t.rvs(df, size=(n_sims, n_assets))


# %%
correlated_shocks_t = z_t @ L.T


# %%
returns_simulated_t = correlated_shocks_t + mu.values


# %%
portfolio_returns_t = returns_simulated_t @ weights


# %%
np.percentile(portfolio_returns, 5), np.percentile(portfolio_returns_t, 5)


# %%
plt.figure(figsize=(10,6))

plt.hist(portfolio_returns, bins=80, density=True, alpha=0.6, label="Gaussian")
plt.hist(portfolio_returns_t, bins=80, density=True, alpha=0.6, label="Student-t")

plt.axvline(np.percentile(portfolio_returns, 5), color="blue", linestyle="--", label="Gaussian 5% VaR")
plt.axvline(np.percentile(portfolio_returns_t, 5), color="red", linestyle="--", label="Student-t 5% VaR")

plt.title("Monte Carlo Portfolio Returns: Gaussian vs Fat-Tailed")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.show()


# %%
plt.savefig(
    "outputs/figures/gaussian_vs_student_t_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


# %%
def ecdf(x):
    return np.sort(x), np.arange(1, len(x)+1)/len(x)

x_g, y_g = ecdf(portfolio_returns)
x_t, y_t = ecdf(portfolio_returns_t)

plt.figure(figsize=(10,6))
plt.plot(x_g, y_g, label="Gaussian")
plt.plot(x_t, y_t, label="Student-t")

plt.title("Empirical CDF: Portfolio Returns")
plt.xlabel("Return")
plt.ylabel("Probability")
plt.legend()
plt.show()


# %%
plt.savefig(
    "outputs/figures/empirical_cdf_comparison.png",
    dpi=300,
    bbox_inches="tight"
)



# %%
plt.figure(figsize=(10,6))

plt.hist(portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 10)],
         bins=50, density=True, alpha=0.6, label="Gaussian Tail")

plt.hist(portfolio_returns_t[portfolio_returns_t < np.percentile(portfolio_returns_t, 10)],
         bins=50, density=True, alpha=0.6, label="Student-t Tail")

plt.title("Left-Tail Comparison (Worst 10%)")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.show()


# %%
plt.savefig(
    "outputs/figures/left_tail_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# %%



