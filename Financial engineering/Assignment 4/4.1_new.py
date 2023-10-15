import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S = 1000
r = 0.02
delta = 0.01
sigma = 0.3
tau = 3/12
K = 1100
alpha = 0.75
u = np.arange(0,100.1, 0.1)
s = np.log(S)

#1
def CharFunc(u):
    cf = np.exp(1j * u * (s + (r - delta - 1/2 * sigma**2) * tau ) - 1/2 * u**2 * sigma**2 * tau)
    return cf

def chi(u):
    chi = np.exp(-r * tau) * CharFunc(u - (1+alpha) * 1j) / (alpha**2 + alpha - u**2 +1j *(1 + 2 * alpha) * u)
    return chi

def integrand(u):
    integrand = np.real(np.exp(-1j * u * np.log(K)) * chi(u))
    return integrand

int = integrand(u)
plt.plot(u, int)
plt.xlabel("u")
plt.show()

#2
def Call_price(u, w):
    c = (1/np.pi * np.sum(w * integrand(u))) * np.exp(-alpha * np.log(K))
    return c

#GL
u = np.array([0.06664545,	0.35004118,	0.85538143,	1.57512003,	2.49836057,	3.61111105,	4.89650447,	6.33505605,	7.90495903,	9.58241541,	11.34199674,	13.15702961,	15,	16.84297039,	18.65800326,	20.41758459,	22.09504097,	23.66494395,	25.10349553,	26.38888895,	27.50163943,	28.42487997,	29.14461857,	29.64995882,	29.93335455,])
w = np.array([0.17090698,	0.3953248,	0.61408735,	0.82357044,	1.02057501,	1.20211051,	1.36542393,	1.50803924,	1.62779437,	1.72287389,	1.79183645,	1.83363664,	1.84764081,	1.83363664,	1.79183645,	1.72287389,	1.62779437,	1.50803924,	1.36542393,	1.20211051,	1.02057501,	0.82357044,	0.61408735,	0.3953248,	0.17090698,])
C_GL = Call_price(u,w)

#Trap
u = np.array([0,	1.25,	2.5,	3.75,	5,	6.25,	7.5,	8.75,	10,	11.25,	12.5,	13.75,	15,	16.25,	17.5,	18.75,	20,	21.25,	22.5,	23.75,	25,	26.25,	27.5,	28.75,	30,])
w = np.array([0.625,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	1.25,	0.625,])
C_trap = Call_price(u,w)

#BS
def C_BS():
    d1 = (np.log(S/K) + (r-delta + 1/2 * sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    C = np.exp(-delta*tau)*S*norm.cdf(d1) - np.exp(-r*tau)*K*norm.cdf(d2)
    return C

C_true = C_BS()

print("C_GL = ", C_GL, "\nC_trap = ", C_trap, "\nC_true = ", C_true)