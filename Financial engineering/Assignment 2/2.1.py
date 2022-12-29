import numpy as np
from numpy import sqrt, exp
from BSM import BSM
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

######1.1+1.2######
S = 1000
s = np.log(S)
r = 0.02
delta = 0.01
T = 3/12
sigmaBS = sqrt(0.04)
sigmaL = sqrt(0.01)
sigmaH = sqrt(0.07)

mness = np.arange(-0.2, 0.21, 0.01)
Kv = S*exp(mness)

N = len(Kv)
CallPrice = np.zeros(N)
IV = np.zeros(N)

for n in range(N):
    K = Kv[n]
    CallPrice[n] = 1/2 * BSM.Call(S, K, T, delta, sigmaL, r) + 1/2 * BSM.Call(S, K, T, delta, sigmaH, r)
    sigma = 0.5
    for i in range(1000):
        d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        vega = S * exp(-delta * T) * sqrt(T) * norm.pdf(d1)
        diff = BSM.Call(S, K, T, delta, sigma, r) - CallPrice[n]
        if abs(diff) < 0.00001:
            break
        sigma = sigma - diff / vega
    IV[n] = sigma

print(IV)
print(CallPrice)

#Plotting
IV_BSM = np.ones(N)*sigmaBS
plt.plot(mness, IV)
plt.plot(mness, IV_BSM)
plt.legend(["stoch. vol", "constant vol"], loc="lower right")
plt.xlabel("Moneyness")
plt.show()

######1.3######
logret = np.arange(-0.5, 0.5, 0.01) #points at which log-return distribution is evaluated
densityL = norm((r-delta-1/2*sigmaL**2)*T,(sigmaL*sqrt(T))).pdf(logret)
densityH = norm((r-delta-1/2*sigmaH**2)*T,(sigmaH*sqrt(T))).pdf(logret)
pdfSV = 0.5*densityL + 0.5*densityH  ##the PDF of stochastic vol
pdf_constVol = norm((r-delta-1/2*sigmaBS**2)*T, sigmaBS*sqrt(T)).pdf(logret)
plt.plot(logret, pdfSV)
plt.plot(logret,pdf_constVol)
plt.legend(["stoch. vol", "constant vol"], loc="upper right")
plt.xlabel("log-return")
plt.show()