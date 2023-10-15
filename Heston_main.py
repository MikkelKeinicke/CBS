from SVJ_functions import funcOptionPrices
import numpy as np
from numpy import sqrt, log, exp
import matplotlib.pyplot as plt
from integration_functions import GL_integration

### The Gauss-Legendre integration scheme
integrationPoints = 100
upperBound = 1024
[uv, wgtv] = GL_integration(upperBound, integrationPoints)
settings = {"uv": uv, "wgt": wgtv}

### Model and initial parameters
settings["model"] = "SV"

# initial stock price parameters
S = 100
r = 0.02
delta = 0.01

####### Fixing the maturity dimension and stydying the vol term structure
tauv = np.array([1 / 52, 3 / 12, 6 / 12, 1, 2, 5])  # maturities
forward_prices = exp((r - delta) * tauv) * S  # forward prices
strike = forward_prices  # ATM options
settings["tauv"] = tauv
settings["strike"] = strike.reshape((1, len(tauv)))

###--- Effect of v_t on volatility term structure
kappa = 2
theta = 0.20 ** 2
rho = -0.25
sigma = 0.1  # Easiest to see effect of v_t when sigma is low

parameters = {"S": S, "r": r, "delta": delta, "kappa": kappa, "theta": theta, "rho": rho, "sigma": sigma}

v0 = np.array([0.10 ** 2, 0.20 ** 2, 0.30 ** 2])
P = np.zeros((len(v0), len(tauv)))  # preallocating
IV = np.zeros((len(v0), len(tauv)))  # preallocating
for i in range(len(v0)):
    parameters["v"] = v0[i]
    P[i, :], IV[i, :] = funcOptionPrices(parameters, settings)

plt.subplot(221)
x = tauv.reshape((len(tauv), 1))
plt.plot(x, IV.T)
plt.title('Effect of v(t)')
plt.legend(['$0.10^2$', '$0.20^2$', '$0.30^2$'], loc='lower right')
plt.xlim(0, 5)
plt.ylim(0.1, 0.3)
plt.xlabel('Option maturity')

###--- Effect of kappa on volatility term structure
theta = 0.20 ** 2
rho = -0.25
sigma = 0.1  # Easiest to see effect of v_t when sigma is low
v = 0.10 ** 2

parameters = {"S": S, "r": r, "delta": delta, "theta": theta, "rho": rho, "sigma": sigma, "v": v}

kappa = np.array([1, 2, 4])
P = np.zeros((len(kappa), len(tauv)))  # preallocating
IV = np.zeros((len(kappa), len(tauv)))  # preallocating
for i in range(len(kappa)):
    parameters["kappa"] = kappa[i]
    [P[i, :], IV[i, :]] = funcOptionPrices(parameters, settings)

plt.subplot(222)
x = tauv.reshape((len(tauv), 1))
plt.plot(x, IV.T)
plt.title('Effect of kappa')
plt.legend(['1', '2', '4'], loc='lower right')
plt.xlim(0, 5)
plt.ylim(0.1, 0.3)
plt.xlabel('Option maturity')

####### Fixing the moneyness dimension and stydying the vol smiles
tauv = np.array([1 / 4])  # maturity
mness = np.arange(-0.20, 0.22, 0.02)  # moneyness
mness = mness.reshape((len(mness), 1))  # column vector
forward_price = exp((r - delta) * tauv) * S
strike = forward_price * exp(mness)
settings["tauv"] = tauv.reshape((1, 1))
settings["strike"] = strike

###--- Effect of rho on volatility smile
kappa = 2
theta = 0.20 ** 2
sigma = 1.00  # Easiest to see effect of rho when sigma is low
v = 0.20 ** 2

parameters = {"S": S, "r": r, "delta": delta, "kappa": kappa, "theta": theta, "sigma": sigma, "v": v}

rho = np.array([-0.5, 0, 0.5])
P = np.zeros((len(mness), len(rho)))  # preallocating
IV = np.zeros((len(mness), len(rho)))  # preallocating
for i in range(len(rho)):
    parameters["rho"] = rho[i]
    [tmp1, tmp2] = funcOptionPrices(parameters, settings)
    P[:, i] = tmp1[:, 0]
    IV[:, i] = tmp2[:, 0]

plt.subplot(223)
x = mness
plt.plot(x, IV)
plt.title('Effect of rho')
plt.legend(['-0.5', '0', '0.5'], loc='lower right')
plt.xlim(-0.22, 0.22)
plt.xlabel('Moneyness')

###--- Effect of sigma on volatility smile
kappa = 2
theta = 0.20 ** 2
rho = -0.25
v = 0.20 ** 2

parameters = {"S": S, "r": r, "delta": delta, "kappa": kappa, "theta": theta, "rho": rho, "v": v}

sigma = np.array([0.2, 0.5, 1, 2])
P = np.zeros((len(mness), len(sigma)))  # preallocating
IV = np.zeros((len(mness), len(sigma)))  # preallocating
for i in range(len(sigma)):
    parameters["sigma"] = sigma[i]
    [tmp1, tmp2] = funcOptionPrices(parameters, settings)
    P[:, i] = tmp1[:, 0]
    IV[:, i] = tmp2[:, 0]

plt.subplot(224)
x = mness
plt.plot(x, IV)
plt.title('Effect of sigma')
plt.legend(['0.2', '0.5', '1', '2'], loc='lower right')
plt.xlim(-0.22, 0.22)
plt.xlabel('Moneyness')

plt.show()
