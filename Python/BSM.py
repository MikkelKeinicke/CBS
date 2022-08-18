import numpy as np
import math
from scipy.stats import norm

"""S = 120
K = 90
T = 5
sigma = 0.5
r = 0.05

from BSM import BSM
print(BSM.Call(S, K, T, sigma, r))
print(BSM.Put(S, K, T, sigma, r))"""

class BSM:

    def d1(S, K, T, sigma, r):
        d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        return d1

    def d2(S, K, T, sigma, r):
        d2 = (np.log(S/K)+(r-sigma**2/2)*T)/(sigma*np.sqrt(T))
        return d2

    def Call(S, K, T, sigma, r):
        d11 = BSM.d1(S, K, T, sigma, r)
        d22 = BSM.d2(S, K, T, sigma, r)
        c = S*norm.cdf(d11)-K*math.exp(-r*T)*norm.cdf(d22)
        return c

    def Put(S, K, T, sigma, r):
        d11 = BSM.d1(S, K, T, sigma, r)
        d22 = BSM.d2(S, K, T, sigma, r)
        p = K*math.exp(-r*T)*norm.cdf(-d22)-S*norm.cdf(-d11)
        return p

#print(Call(120, 90, 5, 0.5, 0.05))
#print(Put(120, 90, 5, 0.5, 0.05))

#list = [2,3,5,3,2,3,425,4,5]

#for i in list:
#    print(Call(i, 90, 5, 0.5, 0.05))


