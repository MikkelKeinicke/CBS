import numpy as np
import matplotlib.pyplot as plt
mu = 0.005
sigma = 0.05
num_sim = 10000
months = 10*12
dt = 1/12
all_years = np.arange(1, months + 1, 1) * dt

log_returns = np.random.normal(loc=mu, scale=sigma, size=(num_sim, months))
cum_log_returns = np.cumsum(log_returns, axis=1)
percentiles = np.percentile(cum_log_returns, [5,95], axis=0)

plt.plot(all_years, cum_log_returns.T, color="grey", alpha=0.5)
plt.plot(all_years, percentiles.T, color="red", linestyle ="--")
plt.grid()
plt.xlabel("Years")
plt.ylabel("Cumulative log-returns")
plt.show()

#Question 2
sigma = np.std(cum_log_returns[:, -1])
sigmaann = np.sqrt(months) * sigma
print(sigma)
print(sigmaann)