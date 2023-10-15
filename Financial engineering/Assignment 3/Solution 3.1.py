from BSM_functions import BSM, BSM_IV
from SVJ_func import mertonJumpFormula
import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

S = 1000
s = log(S)
r = 0.02
delta = 0.01
sigma = 0.10
lamb = 2
muJ = -0.05
sigmaJ = 0.10
sigma_const = sqrt(sigma**2+lamb*((exp(2*muJ+2*sigmaJ**2)-2*exp(muJ+1/2*sigmaJ**2)+1)))
optionType = "call"
parameters = {"S": S, "r": r, "delta": delta, "sigma": sigma, "lamb": lamb, "muJ": muJ, "sigmaJ": sigmaJ}


### Question 1: Jump probabilities
tau = 3/12
res = np.zeros(4)
res[0] = poisson.pmf(k=0, mu=lamb*tau)
res[1] = poisson.pmf(k=1, mu=lamb*tau)
res[2] = poisson.pmf(k=2, mu=lamb*tau)
res[3] = 1-poisson.cdf(k=2, mu=lamb*tau)
print(res)


### Question 2 and 3: Computing prices and implied vol smile
tauv = np.array([1 / 12, 3 / 12, 1/2, 1])  # maturities
mness = np.arange(-0.2, 0.21, 0.01)
Kv = S*exp(mness)
N = len(tauv)  # number of maturities
M = len(Kv)  # number of strikes
P = np.zeros([M,N])  # preallocating
IV = np.zeros([M,N])  # preallocating
for n in range(N):  # one strike at a time
    tau = tauv[n]
    for m in range(M):  # one maturity at a time
        K = Kv[m]
        settings = {"strike": K, "tauv": tau, "truncation": 10}
        P[m,n] = mertonJumpFormula(parameters, settings)
        sigma0 = 0.50  # initial guess
        IV[m,n] = BSM_IV(S, K, delta, r, sigma0, tau, optionType, P[m,n], 0.00001, 1000, "N")

print("HER", P)

x = mness.reshape((len(mness), 1))
plt.plot(x, IV)
plt.plot(x, sigma_const*np.ones([M,1]))
plt.legend(['1 month','3 months','6 months','1 year','Black-Scholes'], loc='upper right')
plt.xlabel('Moneyness')
plt.show()


### Question 4: Compute risk-neutral densities
tau = 1/4
ret = np.arange(-0.5, 0.51, 0.01)  # points at which integrand is evaluated
truncation = 10
mu_bar = exp(muJ + 1 / 2 * sigmaJ ** 2) - 1
pdf_Merton = 0
for k in range(truncation):
    sigma_k = sqrt(sigma ** 2 + k / tau * sigmaJ ** 2)
    r_k = r - mu_bar * lamb + k / tau * (muJ + 1 / 2 * sigmaJ ** 2)
    prob = exp(-lamb * tau) * (lamb * tau) ** k / np.math.factorial(k)
    pdf_Merton = pdf_Merton+prob*norm((r_k-delta-1/2*sigma_k**2)*tau,sigma_k*sqrt(tau)).pdf(ret)

pdf_constVol = norm((r-delta-1/2*sigma_const**2)*tau,sigma_const*sqrt(tau)).pdf(ret)
plt.plot(ret,pdf_Merton)
plt.plot(ret,pdf_constVol)
plt.legend(['Merton model','constant vol'], loc='upper right')
plt.xlabel('log-return')
plt.show()

