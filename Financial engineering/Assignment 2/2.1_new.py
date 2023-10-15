import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

S = 1000
r = 0.02
delta = 0.01
sigmaL = np.sqrt(0.01)
sigmaH = np.sqrt(0.07)
sigma_fix = np.sqrt(0.04)

def C_BS(K, sigma):
    d1 = (np.log(S/K) + (r-delta+1/2*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-delta*T) * S * norm.cdf(d1) - np.exp(-r*T) * K * norm.cdf(d2)

#1.1
T = 3/12
mness = np.arange(-0.2, 0.21, 0.01)
K = np.exp(mness) * S
C = 1/2 * C_BS(K, sigmaL) + 1/2 * C_BS(K,sigmaH)
print("Call Prices =", C)

#1.2
IV = np.zeros_like(C)

for i in range(len(IV)):

    sigma = 0.5
    for n in range(1000):
        d1 = (np.log(S / K[i]) + (r - delta + 1/2 * sigma**2) * T)/(sigma * np.sqrt(T))
        vega = np.exp(-delta * T) * np.sqrt(T) * S * norm.pdf(d1)
        diff = C_BS(K[i], sigma) - C[i]
        if abs(diff)<0.00001:
            break
        sigma = sigma - diff/vega
    IV[i] = sigma

sigma_fix = np.ones_like(IV) * sigma_fix
"""
plt.plot(mness, IV)
plt.plot(mness, sigma_fix)
plt.xlabel("Moneyness")
plt.ylabel("Volatility")
plt.legend(["Stochastic Volatility", "Constant Volatility"], loc="upper center")
plt.show()
"""
#2.3
sigma_fix = np.sqrt(0.04)
logret = np.arange(-0.5, 0.5, 0.01) #points at which log-return distribution is evaluated
densityL = norm((r-delta-1/2*sigmaL**2)*T,(sigmaL*np.sqrt(T))).pdf(logret)
densityH = norm((r-delta-1/2*sigmaH**2)*T,(sigmaH*np.sqrt(T))).pdf(logret)
pdfSV = 0.5*densityL + 0.5*densityH  ##the PDF of stochastic vol

pdf_constVol = norm((r-delta-1/2*sigma_fix**2)*T, sigma_fix*np.sqrt(T)).pdf(logret)

plt.plot(logret, pdfSV)
plt.plot(logret, pdf_constVol)
plt.legend(["stoch. vol", "constant vol"], loc="upper right")
plt.xlabel("log-return")
plt.show()