from BSM_functions import BSM, BSM_IV
import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

S = 1000
s = log(S)
r = 0.02
delta = 0.01
optionType = "call"
sigma = 0.20
sigma_L = sqrt(0.01)
sigma_H = sqrt(0.07)
E_sigma = 0.5*sigma_L+0.5*sigma_H
print(E_sigma)

tau = 1 / 4
mness = np.arange(-0.2, 0.21, 0.01)  # points at which integrand is evaluated
Kv = S*exp(mness)


### Question 1 and 2: Computing prices and implied vol smile
N = len(Kv)  # number of options
P = np.zeros(N)  # preallocating
IV = np.zeros(N)  # preallocating
for n in range(N):  # one strike at a time
    K = Kv[n]
    BS_L = BSM(S, K, delta, r, sigma_L, tau, optionType)
    BS_H = BSM(S, K, delta, r, sigma_H, tau, optionType)
    P[n] = 0.5*BS_L+0.5*BS_H
    sigma0 = 0.50  # initial guess
    IV[n] = BSM_IV(S, K, delta, r, sigma0, tau, optionType, P[n], 0.00001, 1000, "N")

# Plotting smile
IV_constVol = np.ones(N)*sigma
plt.plot(mness, IV)
plt.plot(mness, IV_constVol)
plt.legend(['stoch. vol','constant vol'], loc='lower right')
plt.xlabel('Moneyness')
plt.show()


### Question 3: Compute risk-neutral densities
ret = np.arange(-0.5, 0.51, 0.01)  # points at which integrand is evaluated
pdf_L = norm((r-delta-1/2*sigma_L**2)*tau,sigma_L*sqrt(tau)).pdf(ret)
pdf_H = norm((r-delta-1/2*sigma_H**2)*tau,sigma_H*sqrt(tau)).pdf(ret)
pdf_SV = 0.5*pdf_L+0.5*pdf_H
pdf_constVol = norm((r-delta-1/2*sigma**2)*tau,sigma*sqrt(tau)).pdf(ret)
plt.plot(ret,pdf_SV)
plt.plot(ret,pdf_constVol)
plt.legend(['stoch. vol','constant vol'], loc='upper right')
plt.xlabel('log-return')
plt.show()


### Additional plot
mness = np.arange(-0.2, 0.21, 0.2)
probv = np.arange(0, 1.01, 0.01)
Kv = S*exp(mness)

M = len(probv)  # number of options
N = len(Kv)  # number of options
P = np.zeros([M,N])  # preallocating
IV = np.zeros([M,N])  # preallocating
for n in range(N):  # one strike at a time
    K = Kv[n]
    for m in range(M):  # one strike at a time
        prob = probv[m]
        BS_L = BSM(S, K, delta, r, sigma_L, tau, optionType)
        BS_H = BSM(S, K, delta, r, sigma_H, tau, optionType)
        P[m,n] = (1-prob)*BS_L+prob*BS_H
        sigma0 = 0.50  # initial guess
        IV[m,n] = BSM_IV(S, K, delta, r, sigma0, tau, optionType, P[m,n], 0.00001, 1000, "N")

E_sigma = sqrt((1-probv)*sigma_L**2+probv*sigma_H**2)
x = probv.reshape((len(probv), 1))
plt.plot(x, IV)
plt.plot(x, E_sigma)
plt.xlabel('probability')
plt.show()
