import numpy as np
from numpy import sqrt, log, exp
from scipy.stats import norm


def BSM(S, K, delta, r, sigma, T, optionType):
    # --------------------------------------------------------------------------
    # Computes BSM option prices
    #
    # S: underlying
    # K: strike
    # delta: dividend yield
    # r: risk-free rate
    # sigma: diffusion parameter
    # T: expiration
    # optionType = 'call' or 'put'
    # --------------------------------------------------------------------------
    d1 = (log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if optionType == "call":
        res = S * exp(-delta * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        res = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-delta * T) * norm.cdf(-d1)
    return res


def BSM_vega(S, K, delta, r, sigma, T):
    # --------------------------------------------------------------------------
    # Computes BSM vega
    #
    # S: underlying
    # K: strike
    # delta: dividend yield
    # r: risk-free rate
    # sigma: diffusion parameter
    # T: expiration
    # --------------------------------------------------------------------------
    d1 = (log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    res = S * exp(-delta * T) * sqrt(T) * norm.pdf(d1)
    return res


def BSM_IV(S, K, delta, r, sigma, T, optionType, truePrice, tol, max_iterations, printOutput):
    # --------------------------------------------------------------------------
    # Computes BSM implied volatility
    #
    # S: underlying
    # K: strike
    # delta: dividend yield
    # r: risk-free rate
    # sigma: diffusion parameter
    # T: expiration
    # optionType = 'call' or 'put'
    # --------------------------------------------------------------------------
    for i in range(max_iterations):

        # calculate difference between BSM price and market price with
        # iteratively updated volatility estimate
        diff = BSM(S, K, delta, r, sigma, T, optionType) - truePrice

        # break if difference is less than specified tolerance level
        if abs(diff) < tol:
            break

        # use newton-raphson to update the estimate
        sigma = sigma - diff / BSM_vega(S, K, delta, r, sigma, T)
    if printOutput == "Y":
        print(f'found on {i}th iteration')
        print(f'difference is equal to {diff}')

    return sigma
