import numpy as np
import math
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
from BSM import BSM

sigma = 0.1
lamb = 2
muJ = -0.05
sigmaJ = 0.1
sigmaBS = 0.1796

S = 1000
r = 0.02
delta = 0.01
Tau = 3/12
sigma_const = np.sqrt(sigma**2+lamb*((np.exp(2*muJ+2*sigmaJ**2)-2*np.exp(muJ+1/2*sigmaJ**2)+1))) #????

#1
print(2*3/12)
#Probabilities
def possion(k):
    prob = ((np.exp(-2 * Tau) * 2**k * Tau**k) / math.factorial(k))
    return prob
print(possion(0))
print(possion(1))
print(possion(2))
print(1-possion(0)-possion(1)-possion(2))

#alternative:
print(poisson.pmf(k=0, mu=lamb*Tau))
print(poisson.pmf(k=1, mu=lamb*Tau))
print(poisson.pmf(k=2, mu=lamb*Tau))

#2 & 3
#define the MertonJump
def MertonJump(S, sigma, lamb, muJ, sigmaJ, r, delta, K, Tau, truncation):
    from BSM import BSM
    mu_bar = np.exp(muJ + 1/2 * sigmaJ**2)-1
    lamb_prime = lamb * (1 + mu_bar)

    sum = 0     #start the sum
    for k in range(truncation):
        sigma_k = np.sqrt(sigma ** 2 + k / Tau * sigmaJ ** 2)
        r_k = r - mu_bar * lamb + k / Tau * (muJ + 1/2 * sigmaJ**2)
        prob = ((np.exp(-lamb_prime * Tau) * lamb_prime**k * Tau**k) / math.factorial(k))

        sum = sum + prob * BSM.Call(S, K, Tau, delta, sigma_k, r_k)
        print(sum)
    return sum

mness = np.arange(-0.2, 0.21, 0.01)
tauv = np.array([1/12, 3/12, 6/12, 12/12])
Kv = S*np.exp(mness)
N = len(tauv)  #number of moneyness
M = len(Kv)    #number of strikes

P = np.zeros([M, N])
IV = np.zeros([M, N])

#calculate prices and IV
for n in range(N):  #loop over maturities
    tau = tauv[n]
    for m in range(M):  #loop over strikes
        K = Kv[m]
        truncation = 10  # i.e., we start with 0 to 10 jumps
        P[m, n] = MertonJump(S, sigma, lamb, muJ, sigmaJ, r, delta, K, tau, truncation)

        #IV
        sigma0 = 0.4
        for i in range(1000):
            d1 = (np.log(S / K) + (r - delta + 0.5 * sigma0 ** 2) * tau) / (sigma0 * np.sqrt(tau))
            vega = S * np.exp(-delta * tau) * np.sqrt(tau) * norm.pdf(d1)
            diff = BSM.Call(S, K, tau, delta, sigma0, r) - P[m, n]
            if abs(diff) < 0.00001:
                break
            sigma0 = sigma0 - diff / vega
        IV[m, n] = sigma0

x = mness.reshape((len(mness), 1))
plt.plot(x, IV)
plt.plot(x, sigma_const*np.ones([M,1]))
plt.legend(["1 month", "3 months", "6 months", "1 year", "Black-Scholes"], loc="upper right")
plt.xlabel("Moneyness")
plt.show()

#4
tau = 1/4
ret = np.arange(-0.5, 0.51, 0.01)   #integrand evaluated at

truncation = 10
mu_bar = np.exp(muJ + 1/2 * sigmaJ**2)-1

pdf_Merton = 0
for k in range(truncation):
    sigma_k = np.sqrt(sigma ** 2 + k / Tau * sigmaJ ** 2)
    r_k = r - mu_bar * lamb + k / Tau * (muJ + 1 / 2 * sigmaJ ** 2)
    prob = ((np.exp(-lamb * Tau) * lamb ** k * Tau ** k) / math.factorial(k))
    pdf_Merton = pdf_Merton + prob * norm((r_k-delta-1/2*sigma_k**2)*tau,sigma_k*np.sqrt(tau)).pdf(ret)

pdf_constVol = norm((r-delta-1/2*sigma_const**2)*tau,sigma_const*np.sqrt(tau)).pdf(ret)
plt.plot(ret, pdf_Merton)
plt.plot(ret, pdf_constVol)
plt.legend(["Merton model", "constant vol"], loc="upper right")
plt.xlabel("log-return")
plt.show()


















