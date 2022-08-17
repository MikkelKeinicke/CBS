import numpy as np
import math
from scipy.stats import norm

def d1(S, K, T, sigma, r):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    return d1

def d2(S, K, T, sigma, r):
    d2 = (np.log(S/K)+(r-sigma**2/2)*T)/(sigma*np.sqrt(T))
    return d2

def Call(S, K, T, sigma, r):
    d11 = d1(S, K, T, sigma, r)
    d22 = d2(S, K, T, sigma, r)
    c = S*norm.cdf(d11)-K*math.exp(-r*T)*norm.cdf(d22)
    return c

def Put(S, K, T, sigma, r):
    d11 = d1(S, K, T, sigma, r)
    d22 = d2(S, K, T, sigma, r)
    p = K*math.exp(-r*T)*norm.cdf(-d22)-S*norm.cdf(-d11)
    return p

print(Call(110, 90, 5, 0.5, 0.05))
print(Put(110, 90, 5, 0.5, 0.05))