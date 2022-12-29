from BSM_functions import BSM, BSM_IV
import numpy as np
from numpy import sqrt, log, exp, pi, real


def charFuncSV(u, tau, parameters):
    # --------------------------------------------------------------------------
    # Computes characteristic functions for SV model
    # --------------------------------------------------------------------------
    x = log(parameters["S"])
    kappa = parameters["kappa"]
    theta = parameters["theta"]
    rho = parameters["rho"]
    sigma = parameters["sigma"]
    v = parameters["v"]
    r = parameters["r"]
    delta = parameters["delta"]

    d = sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma ** 2 * (u * 1j + u ** 2))
    d = -d  # That little Heston trap
    g = (kappa - rho * sigma * u * 1j + d) / (kappa - rho * sigma * u * 1j - d)
    C = (r - delta) * u * 1j * tau + kappa * theta / sigma ** 2 * (
                (kappa - rho * sigma * u * 1j + d) * tau - 2 * log((1 - g * exp(d * tau)) / (1 - g)))
    D = (kappa - rho * sigma * u * 1j + d) / sigma ** 2 * (1 - exp(d * tau)) / (1 - g * exp(d * tau))
    out = exp(C + D * v + 1j * u * x)
    return out


def charFuncJ(u, tau, parameters):
    # --------------------------------------------------------------------------
    # Computes characteristic functions for J model
    # --------------------------------------------------------------------------
    x = log(parameters["S"])
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    r = parameters["r"]
    delta = parameters["delta"]

    C = (1j * u * (r - delta) - 1 / 2 * u * (u + 1j) * sigma ** 2 + lamb * (
                (exp(1j * u * muJ - 1 / 2 * u ** 2 * sigmaJ ** 2) - 1) - 1j * u * (
                    exp(muJ + 1 / 2 * sigmaJ ** 2) - 1))) * tau
    out = exp(C + 1j * u * x)
    return out


def charFuncSVJ(u, tau, parameters):
    # --------------------------------------------------------------------------
    # Computes characteristic functions for SVJ model
    # --------------------------------------------------------------------------
    x = log(parameters["S"])
    kappa = parameters["kappa"]
    theta = parameters["theta"]
    rho = parameters["rho"]
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    v = parameters["v"]
    r = parameters["r"]
    delta = parameters["delta"]

    d = sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma ** 2 * (u * 1j + u ** 2))
    d = -d  # That little Heston trap
    g = (kappa - rho * sigma * u * 1j + d) / (kappa - rho * sigma * u * 1j - d)
    C = (r - delta) * u * 1j * tau + kappa * theta / sigma ** 2 * (
                (kappa - rho * sigma * u * 1j + d) * tau - 2 * log((1 - g * exp(d * tau)) / (1 - g)))
    D = (kappa - rho * sigma * u * 1j + d) / sigma ** 2 * (1 - exp(d * tau)) / (1 - g * exp(d * tau))
    Cmerton = (lamb * (exp(1j * u * muJ - 1 / 2 * u ** 2 * sigmaJ ** 2) - 1 - 1j * u * (
                exp(muJ + 1 / 2 * sigmaJ ** 2) - 1))) * tau
    out = exp(C + D * v + 1j * u * x + Cmerton)
    return out


def mertonJumpFormula(parameters, settings):
    # --------------------------------------------------------------------------
    # Computes prices and implied vols for J model using Merton(1976) formula
    # --------------------------------------------------------------------------
    S = parameters["S"]
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    r = parameters["r"]
    delta = parameters["delta"]

    K = settings["strike"]
    tau = settings["tauv"]
    truncation = settings["truncation"]

    mu_bar = exp(muJ + 1 / 2 * sigmaJ ** 2) - 1
    lamb_prime = lamb * (1 + mu_bar)
    tmp = 0
    for k in range(truncation):
        sigma_k = sqrt(sigma ** 2 + k / tau * sigmaJ ** 2)
        r_k = r - mu_bar * lamb + k / tau * (muJ + 1 / 2 * sigmaJ ** 2)
        prob = exp(-lamb_prime * tau) * (lamb_prime * tau) ** k / np.math.factorial(k)
        tmp = tmp + prob * BSM(S, K, delta, r_k, sigma_k, tau, "call")
        print(tmp)
    return tmp


def funcOptionPrices(parameters, settings):
    # --------------------------------------------------------------------------
    # Computes prices and implied vols for SV, J, and SVJ models
    # --------------------------------------------------------------------------
    S = parameters["S"]
    r = parameters["r"]
    delta = parameters["delta"]

    uv = settings["uv"]
    wgt = settings["wgt"]
    strike = settings["strike"]
    [N, M] = strike.shape  # N = number of strikes, M = number of option maturities
    tauv = settings["tauv"]
    model = settings["model"]
    price_fit = np.zeros((N, M))  # preallocating
    IV_fit = np.zeros((N, M))  # preallocating
    for m in range(M):  # one option maturity at a time
        tau = tauv[m]  # option maturity
        ###Compute characteristic function (depoends on maturity, NOT on strike)
        if model == "SV":
            charFct1 = charFuncSV(uv - 1j, tau, parameters) / (S * exp((r - delta) * tau))  # psi in pi2
            charFct2 = charFuncSV(uv, tau, parameters)  # psi in pi2
        if model == "J":
            charFct1 = charFuncJ(uv - 1j, tau, parameters) / (S * exp((r - delta) * tau))  # psi in pi2
            charFct2 = charFuncJ(uv, tau, parameters)  # psi in pi2
        if model == "SVJ":
            charFct1 = charFuncSVJ(uv - 1j, tau, parameters) / (S * exp((r - delta) * tau))  # psi in pi2
            charFct2 = charFuncSVJ(uv, tau, parameters)  # psi in pi2
        ###Compute call prices
        for n in range(N):  # one strike at a time
            K = strike[n, m]  # strike
            # call price
            integrand1 = real(exp(-1j * uv * log(K)) * charFct1 / (1j * uv))  # integrand in pi1
            integral = sum(wgt * integrand1)  # computing integral by numerical integration
            P1 = 1 / 2 + 1 / pi * integral
            integrand2 = real(exp(-1j * uv * log(K)) * charFct2 / (1j * uv))  # integrand in pi2
            integral = sum(wgt * integrand2)  # computing integral by numerical integration
            P2 = 1 / 2 + 1 / pi * integral
            price = S * exp(-delta * tau) * P1 - K * exp(-r * tau) * P2  # call price

            # implied volatility
            sigma0 = 0.20  # initial guess
            IV = BSM_IV(S, K, delta, r, sigma0, tau, "call", price, 0.00001, 1000, "N")

            price_fit[n, m] = price
            IV_fit[n, m] = IV

    return price_fit, IV_fit
