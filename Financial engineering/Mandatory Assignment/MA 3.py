import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

# Parameters
r = 0.023
delta = 0.017
S = 3419
K = 3500
dt = 1 / 252

tau = 3 / 12  # for new sim, change.
steps = 63  # for new sim, change: steps = tau * 252
# when changing, they need to match in dimensions. I.e., steps = tau in days.
# tau = 1, steps = 252

NoPrices = steps + 1
paths = 50000

kappa = 3.0
theta = 0.04
rho = -0.4
sigma = 0.3
v = 0.03

#Simulate from Heston
def HestonSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, return_prices = False, return_variances = False, return_log_prices = False):
    size = (paths, NoPrices)
    prices = np.zeros(size)
    variances = np.zeros(size)
    log_prices = np.zeros(size)

    mean = np.array([0,0])       #mean 0
    cov = np.array([[1, rho],    #corr = rho,
                   [rho, 1]])    #variance = 1
    S_t = S
    v_t = v
    V_t = v

    prices[:,0] = S
    variances[:,0] = v
    log_prices[:,0] = np.log(S)

    zeros = np.zeros(len(prices))

    for t in range(steps):
        W_t = np.random.multivariate_normal(mean, cov, size = paths) * np.sqrt(dt)   #Generate Wiener

        S_t = S_t * (np.exp((r - 0.5 * V_t) * dt + np.sqrt(V_t) * W_t[:,0]))
        s = np.log(S_t)
        v_t = v_t + kappa * ( theta - V_t) * dt + sigma * np.sqrt(V_t) * W_t[:,1]

        V_t = np.maximum(zeros, v_t)    #fixes variance going negative / sqrt of negative number

        prices[:,t+1] = S_t
        variances[:,t+1] = v_t
        log_prices[:,t+1] = s

    if return_variances:
        return variances
    if return_prices:
        return prices
    if return_log_prices:
        return log_prices

sim = HestonSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, return_prices = True, return_variances = False, return_log_prices = False)
x = np.arange(0, tau+dt, dt)

plt.plot(x, sim.T, color="grey")
plt.grid()
plt.xlabel("Time (Years)")
plt.ylabel("Log-price")
plt.show()

#3.1
sim = HestonSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, return_prices = True, return_variances = False, return_log_prices = False)
A = len(sim.T)-1
sim = sim[:,A]  #extract the final log.prices

#sum stats
sum_stat = np.zeros(6)

sum_stat[0] = np.average(sim)
sum_stat[1] = np.std(sim)
sum_stat[2] = np.var(sim)
sum_stat[3] = np.median(sim)
sum_stat[4] = skew(sim)
sum_stat[5] = kurtosis(sim, fisher=False)   #fisher: True = excess
print(sum_stat)

#generate normal-dist with same mean, var
binwidth = 0.005   #variances = 0.001, prices = 0.005

normS = np.arange(7.6, 8.6, 0.01)
norm_dens = norm(np.average(sim), np.std(sim)).pdf(normS)

plt.hist(sim, bins = np.arange(min(sim), max(sim) + binwidth, binwidth), density=True)

plt.plot(normS, norm_dens)
plt.legend(["Normal distribution","Log-Prices density"])
plt.xlabel("Log-Prices")
plt.show()

# define basis functions
constFunc = lambda *args, **kwargs: np.ones_like(kwargs['St'])
moneyness_t = lambda St, K, *args, **kwargs: St / K
moneyness_t_squared = lambda St, K, *args, **kwargs: (St / K) ** 2
moneyness_t_qubed = lambda St, K, *args, **kwargs: (St / K) ** 3
var_t = lambda vt, *args, **kwargs: vt
var_t_squared = lambda vt, *args, **kwargs: vt ** 2
var_t_qubed = lambda vt, *args, **kwargs: vt ** 3
moneyVar = lambda St, K, vt, *args, **kwargs: vt * (St / K)
moneyVar_squared = lambda St, K, vt, *args, **kwargs: (vt ** 2) * (St / K)
money_squared_Var = lambda St, K, vt, *args, **kwargs: vt * ((St / K) ** 2)
moneyness_t_forthed = lambda St, K, *args, **kwargs: (St / K) ** 4
var_t_forthed = lambda vt, *args, **kwargs: vt ** 4

set1 = [constFunc, moneyness_t, moneyness_t_squared,
        var_t, var_t_squared]
set2 = [constFunc, moneyness_t, moneyness_t_squared, moneyness_t_qubed,
        var_t, var_t_squared, var_t_qubed]
set3 = [constFunc, moneyness_t, moneyness_t_squared, moneyness_t_qubed,
        var_t, var_t_squared, var_t_qubed, moneyVar]
set4 = [constFunc, moneyness_t, moneyness_t_squared, moneyness_t_qubed,
        var_t, var_t_squared, var_t_qubed, moneyVar, moneyVar_squared, money_squared_Var]
set5 = [constFunc, moneyness_t, moneyness_t_squared, moneyness_t_qubed, moneyness_t_forthed,
        var_t, var_t_squared, var_t_qubed, var_t_forthed, moneyVar, moneyVar_squared, money_squared_Var]

# Calculate price
def HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, funcset, return_American=False,
                   return_European=False):
    np.random.seed(57781)
    size = (paths * 2, NoPrices)  # *2 because using antithetic

    StockPrices = np.zeros(size)
    Variances = np.zeros(size)

    df = np.exp(-r * dt)  # discount factor

    mean = np.array([0, 0])  # mean 0
    cov = np.array([[1, rho],  # corr = rho,
                    [rho, 1]])  # variance = 1
    S_t = S
    v_t = v
    V_t = v

    StockPrices[:, 0] = S
    Variances[:, 0] = v

    zeros = np.zeros(paths * 2)

    for t in range(steps):
        W_t = np.random.multivariate_normal(mean, cov, size=paths) * np.sqrt(dt)  # Generate Wiener
        W_t2 = - W_t  # Antithetic
        W_t = np.vstack((W_t, W_t2))

        S_t = S_t * np.exp((r - delta - 0.5 * V_t) * dt + np.sqrt(V_t) * W_t[:, 0])
        v_t = v_t + kappa * (theta - V_t) * dt + sigma * np.sqrt(V_t) * W_t[:, 1]

        V_t = np.maximum(zeros, v_t)  # fixes variance going negative / sqrt of negative number

        StockPrices[:, t + 1] = S_t
        Variances[:, t + 1] = V_t

    p = np.maximum(K - StockPrices, 0)  # put value in every matrix
    OptValue = np.zeros_like(p)  # matrix for option values
    OptValue[:, -1] = p[:, -1]  # Put value at maturity equal to K-S

    EuropeanPut = np.sum(OptValue[:, -1] * np.exp(-r * tau)) / (paths * 2)

    for t in range(steps - 1, 0, -1):  # backwards loop from end

        ITM = p[:, t].nonzero()  # paths that are ITM at time t
        OTM = np.where(p[:, t] == 0)  # paths that are OTM at time t

        OptValue[:, t][OTM] = OptValue[:, t + 1][OTM] * df  # Option value at time t, if P is OTM

        Y = OptValue[:, t + 1][ITM] * df  # Discounted realized cash flow from continuation at time t+1
        St = StockPrices[:, t][ITM]  # Stock Price paths that are ITM at time t this periods stock price for ITM paths
        vt = Variances[:, t][ITM]  # Variances

        X = np.array(list(map(lambda func: func(St=St, vt=vt, K=K), funcset))).T

        reg = sm.OLS(Y, X).fit()  # cross-sectional regression
        ContValue = reg.fittedvalues  # evaluate regression at each loop (Continuation value)

        OptValue[:, t][ITM] = np.where(p[:, t][ITM] > ContValue,  # Option value at time t, if P is ITM
                                       p[:, t][ITM], OptValue[:, t + 1][ITM] * df)  # compare values. p[t,:] if true

    P0 = np.mean(OptValue[:, 1] * df)

    Se = np.std(OptValue[:,1]*df)/np.sqrt(len(OptValue[:,1]))
    ConfD = [P0 - 1.96*Se, P0 + 1.96*Se]

    if return_American:
        return P0
    if return_European:
        return EuropeanPut

LSM_Price1 = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set1, return_American=True,
                            return_European=False)
LSM_Price2 = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set2, return_American=True,
                            return_European=False)
LSM_Price3 = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set3, return_American=True,
                            return_European=False)
LSM_Price4 = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set4, return_American=True,
                            return_European=False)
LSM_Price5 = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set5, return_American=True,
                            return_European=False)
European = HestonPriceSim(S, tau, r, delta, K, kappa, theta, rho, sigma, v, paths, steps, set5, return_American=False,
                          return_European=True)

print(" European Put   = ", European, "\n", "American Put 1 = ", LSM_Price1, "\n", "American Put 2 = ", LSM_Price2,
      "\n", "American Put 3 = ", LSM_Price3, "\n", "American Put 4 = ", LSM_Price4, "\n", "American Put 5 = ",
      LSM_Price5)