import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

# Drawing data
maturity = np.array([30, 61, 93, 183, 365])
df_strike = np.array(
    pd.read_excel(r'C:\Users\Mikke\OneDrive\CBS - FIN\FINANCIAL ENGINEERING\ASSIGNMENTS\MA\SX5E_options.xlsx',
                  sheet_name='options', usecols='B:F', skiprows=2, nrows=13, names=maturity))
df_price = np.array(
    pd.read_excel(r'C:\Users\Mikke\OneDrive\CBS - FIN\FINANCIAL ENGINEERING\ASSIGNMENTS\MA\SX5E_options.xlsx',
                  sheet_name='options', usecols='B:F', skiprows=16, nrows=13, names=maturity))

# Defining variables:
r = 0.023
delta = 0.017
tau = maturity / 365
S = 3419

# Creating IV matrix:
N = df_strike.shape
O = len(df_strike)
P = df_strike.shape[1]
IV = np.zeros(N)
Vega = np.zeros(N)


def BSM_C(S, K, sigma, r, Tau, delta):
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * Tau) / (sigma * np.sqrt(Tau))
    d2 = d1 - sigma * np.sqrt(Tau)
    C = np.exp(-delta * Tau) * S * norm.cdf(d1) - np.exp(-r * Tau) * K * norm.cdf(d2)
    return C

for n in range(P):
    Tau = tau[n]
    for m in range(O):
        K = df_strike[m, n]
        price = df_price[m, n]

        sigma = 0.4
        for i in range(10000):
            C = BSM_C(S, K, sigma, r, Tau, delta)
            d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * Tau) / (sigma * np.sqrt(Tau))
            vega = S * np.exp(-delta * Tau) * np.sqrt(Tau) * norm.pdf(d1)
            diff = C - price
            if abs(diff) <= 0.00001:
                break
            sigma = sigma - diff / vega
        Vega[m, n] = vega
        IV[m, n] = sigma

print("IV=", IV)
print("Vega=", Vega)

# Plotting the implied volatility smile:
Ft = S * np.exp((r-delta) * tau)

for n in range(5):
    Ft = S * np.exp((r-delta) * tau[n])
    mness = np.log(df_strike[:, n] / Ft)
    plt.plot(mness, IV[:, n])
plt.legend(['30 days', '61 days', '93 days', '183 days', '365 days'], loc='upper right')
plt.xlabel("Moneyness")
plt.ylabel("Implied volatility")
plt.show()

# Plotting delta nd vega as a function of moneyness:
for n in range(5):
    Ft = S * np.exp((r-delta) * tau[n])
    mness = np.log(df_strike[:, n] / Ft)
    sigma = IV[:, n]
    d1 = (np.log(S / df_strike[:, n]) + (r - delta + 0.5 * sigma ** 2) * tau[n]) / (sigma * np.sqrt(tau[n]))
    vega = S * np.exp(-delta * tau[n]) * np.sqrt(tau[n]) * norm.pdf(d1)
    Delta = np.exp(-delta * tau[n]) * norm.cdf(d1)
    plt.plot(mness, vega)
plt.legend(['30 days', '61 days', '93 days', '183 days', '365 days'], loc='upper right')
plt.xlabel("Moneyness")
plt.ylabel("Vega")
plt.show()

# Plotting delta nd vega as a function of moneyness:
for n in range(5):
    Ft = S * np.exp((r-delta) * tau[n])
    mness = np.log(df_strike[:, n] / Ft)
    sigma = IV[:, n]
    d1 = (np.log(S / df_strike[:, n]) + (r - delta + 0.5 * sigma ** 2) * tau[n]) / (sigma * np.sqrt(tau[n]))
    Delta = np.exp(-delta * tau[n]) * norm.cdf(d1)
    plt.plot(mness, Delta)
plt.legend(['30 days', '61 days', '93 days', '183 days', '365 days'], loc='upper right')
plt.xlabel("Moneyness")
plt.ylabel("Delta")
plt.show()

################Calibrating Heston model################
def GL_integration(xmax, points):
    [u, wgt] = np.polynomial.legendre.leggauss(points)
    u = u + 1
    u = u * xmax / 2
    wgt = wgt * xmax / 2
    return u, wgt

def CharFunc(u, tau, kappa, theta, rho, sigma, v):

    d = np.sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma ** 2 * (u * 1j + u ** 2))
    d = -d  # Heston trap
    g = (kappa - rho * sigma * u * 1j + d) / (kappa - rho * sigma * u * 1j - d)

    M = (r - delta) * u * 1j * tau + kappa * theta / sigma**2 * ((kappa - rho * sigma * u * 1j + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
    N = (kappa - rho * sigma * u * 1j + d) / sigma ** 2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))

    psi = np.exp(M + N * v + 1j * u * np.log(S))
    return psi

def PriceFunc(kappa, theta, rho, sigma, v):

    [N, M] = df_strike.shape
    price_matrix = np.zeros((N,M))

    points = 100  # number of integration points
    xmax = 2000  # truncation of integral
    [u, wgt] = GL_integration(xmax, points)

    #Each maturity (M=5)
    for m in range(M):
        tau = tauv[m]

        psi1 = CharFunc(u - 1j, tau, kappa, theta, rho, sigma, v)
        psi2 = CharFunc(u, tau, kappa, theta, rho, sigma, v)

        #Each strike (N=13)
        for n in range(N):
            K = df_strike[n,m]

            integrand = np.real(np.exp(-1j * u * np.log(K)) * psi1 / (1j * u * S * np.exp((r - delta) * tau)))
            integral = sum(wgt * integrand)
            pi1 = 1/2 + 1/np.pi * integral

            integrand = np.real(np.exp(-1j * u * np.log(K)) * psi2 / (1j * u))
            integral = sum(wgt * integrand)
            pi2 = 1/2 + 1/np.pi * integral

            HestPrice = S * np.exp(-delta * tau) * pi1 - K * np.exp(-r * tau) * pi2
            price_matrix[n,m] = HestPrice

    return price_matrix

tauv = np.array(tau)

#########Calibrate#########
bounds = ([1e-8,     1e-8,      -1,     1e-8,        1e-8],
          [np.inf,   np.inf,    0,      np.inf,      np.inf])

init_kappa = 3.9
init_theta = 0.02
init_rho = -0.4
init_sigma = 0.6
init_v = 0.01
initguess = [init_kappa, init_theta, init_rho, init_sigma, init_v]

P = df_price

def Err(x):
    kappa, theta, rho, sigma, v = x
    err = np.sum((PriceFunc(kappa, theta, rho, sigma, v)-P)/Vega)  # np.sum will give a scalar value which the function needs
    return err

res = opt.least_squares(Err, initguess, bounds = bounds, verbose = 2)
kappa, theta, rho, sigma, v = res.x

CalibratedPrice = PriceFunc(kappa, theta, rho, sigma, v)
print("\n", "               kappa,  theta,  rho, sigma,      v", "\n", "Parameters = ", res.x, "\n", "\n", "True Price = ", "\n", P, "\n", "\n", "Heston price = ", "\n", CalibratedPrice)

#Fitted implied vol
FitPrice = PriceFunc(kappa, theta, rho, sigma, v)
P = df_strike.shape[1]

IVf = np.zeros(df_strike.shape)

for n in range(P):
    tau = tauv[n]
    for m in range(O):
        K = df_strike[m, n]
        price = FitPrice[m, n]

        for i in range(10000):
            C = BSM_C(S, K, sigma, r, tau, delta)
            d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
            vega = S * np.exp(-delta * tau) * np.sqrt(tau) * norm.pdf(d1)
            diff = C - price
            if abs(diff) <= 0.00001:
                break
            sigma = sigma - diff / vega
        Vega[m, n] = vega
        IVf[m, n] = sigma
print("IV Fitted = ", "\n", IVf)

title = ['Maturity: 30 days', 'Maturity: 61 days', 'Maturity: 93 days', 'Maturity: 183 days', 'Maturity: 365 days']

for n in range(5):
    Ft = S * np.exp((r-delta) * tauv[n])
    mness = np.log(df_strike[:, n] / Ft)

    Title = title[n]

    plt.plot(mness, IV[:, n])
    plt.plot(mness, IVf[:,n])
    plt.title(Title)
    plt.legend(["Actual Implied Volatility", "Fitted Implied Volaility"])
    plt.xlabel("Moneyness")
    plt.ylabel("Implied volatility")
    plt.show()