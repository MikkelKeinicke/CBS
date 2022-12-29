import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def CharFuncBSM(uv):
    ret = np.exp(1j * uv * (s + (r - delta - 1/2 * sigma**2)*Tau) - 1/2 * uv**2 * sigma**2 * Tau)
    return ret

S = 1000
s = np.log(S)
K = 1100
k = np.log(K)
r = 0.02
delta = 0.01
sigma = 0.3
alpha = 0.75
Tau = 3/12

###Question 1
uv = np.arange(0, 100, 0.1)

psi = CharFuncBSM(uv - (1 + alpha) * 1j)
chi = np.exp(-r*Tau) * psi / (alpha**2 + alpha - uv**2 + 1j * (1+2*alpha) * uv)
integrand = np.real(np.exp(-1j * uv * k) * chi)

plt.plot(uv, integrand)
plt.xlabel("u")
plt.show()

###Question 2
def Callprice():
    psi = CharFuncBSM(uv - (1 + alpha) * 1j)
    chi = np.exp(-r * Tau) * psi / (alpha ** 2 + alpha - uv ** 2 + 1j * (1 + 2 * alpha) * uv)
    integrand = np.real(np.exp(-1j * uv * k) * chi)
    integral = sum(wgtv * integrand)
    mod_price = 1/ np.pi * integral
    price = np.exp(-alpha*k)*mod_price
    return price

#Gauss-Legendre price
[uv, wgtv] = np.array([[0.06664545, 0.35004118, 0.85538143, 1.57512003,2.49836057,3.61111105,4.89650447,6.33505605,7.90495903,9.58241541,11.34199674,13.15702961,15,16.84297039,18.65800326,20.41758459,22.09504097,23.66494395,25.10349553,26.38888895,27.50163943,28.42487997,29.14461857,
29.64995882,29.93335455], [0.17090698,0.3953248,0.61408735,0.82357044,1.02057501,1.20211051,1.36542393,1.50803924,1.62779437,1.72287389,1.79183645,1.83363664,1.84764081,1.83363664,1.79183645,1.72287389,1.62779437,1.50803924,1.36542393,1.20211051,1.02057501,0.82357044,0.61408735,0.3953248,0.17090698]])
C_GL = Callprice()
print("C_GL = ", C_GL)

#Trapezoid price
[uv, wgtv] = np.array([[0,1.25,2.5,3.75,5,6.25,7.5,8.75,10,11.25,12.5,13.75,15,16.25,17.5,18.75,20,21.25,22.5,23.75,25,26.25,27.5,28.75,30],[0.625,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,0.625]])
C_Trap = Callprice()
print("C_Trap = ", C_Trap)

#True price
def C_true():
    d1 = (np.log(S/K)+(r-delta+sigma**2/2)*Tau)/(sigma*np.sqrt(Tau))
    d2 = d1 - sigma * np.sqrt(Tau)
    C = np.exp(-delta*Tau) * S * norm.cdf(d1) - np.exp(-r*Tau) * K * norm.cdf(d2)
    return C
C_BSM = C_true()
print("BSM = ", C_BSM)
