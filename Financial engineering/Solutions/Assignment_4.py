from BSM_functions import BSM
import numpy as np
from numpy import sqrt, log, exp, pi, real
import matplotlib.pyplot as plt
from integration_functions import GL_integration, trap_integration


### Question 1: Plotting integrands
def charFuncBSM(uv):
    # Computes characteristic functions for BSM model
    out = exp(1j * uv * (s + (r - delta - 1 / 2 * sigma ** 2) * tau) - 1 / 2 * uv ** 2 * sigma ** 2 * tau)
    return out

S = 1000
s = log(S)
r = 0.02
delta = 0.01
sigma = 0.30
tau = 1 / 4
K = 1100
alpha = 0.75

uv = np.arange(0, 100, 0.1)  # points at which integrand is evaluated

psi = charFuncBSM(uv - (1+alpha)*1j)
chi = exp(-r*tau) * psi / (alpha**2 + alpha - uv**2 + 1j*(1+2*alpha)*uv)
integrand = real(exp(-1j * uv * log(K)) * chi)  # integrand

plt.plot(uv, integrand)
plt.xlabel('u')
plt.show()


### Question 2: Comparison of integration schemes with analytical price
def callPrice():
    psi = charFuncBSM(uv - (1 + alpha) * 1j)
    chi = exp(-r * tau) * psi / (alpha ** 2 + alpha - uv ** 2 + 1j * (1 + 2 * alpha) * uv)
    integrand = real(exp(-1j * uv * log(K)) * chi)  # integrand
    integral = sum(wgtv * integrand)  # computing integral by numerical integration
    price_modified = 1/pi*integral  # modified call price
    price = exp(-alpha*log(K))*price_modified  # call price
    return price

nPoints = 25  # number of integration points
upperBound = 30  # truncation of integral

# Price via the Gauss-Legendre integration scheme
[uv, wgtv] = GL_integration(upperBound, nPoints)
C_GL = callPrice()

# Price via trapezoid scheme
[uv, wgtv] = trap_integration(0, upperBound, nPoints)
C_Trap = callPrice()

# True price
C_true = BSM(S, K, delta, r, sigma, tau, "call")

res = np.array([C_true, C_GL, C_Trap])

print(res)


"""
### Question 3: Convergence to true price
nPointsv = np.linspace(10, 50, 9, dtype='int')
upperBound = 30  # truncation of integral
Cv = np.zeros((len(nPointsv), 3))  # preallocating
Cv[:, 0] = C_true
for i in range(len(nPointsv)):
    nPoints = nPointsv[i]
    # Price via the Gauss-Legendre integration scheme
    [uv, wgtv] = GL_integration(upperBound, nPoints)
    Cv[i, 1] = callPrice()
    # Price via trapezoid scheme
    [uv, wgtv] = trap_integration(0, upperBound, nPoints)
    Cv[i, 2] = callPrice()

print(Cv)

plt.plot(nPointsv, Cv)
plt.gca().legend(('analytical', 'Gauss-Legendre integration', 'trapezoid integration'))
plt.xlabel('Number of integration points')
plt.show()


### Printing integration points for Excel
np.set_printoptions(precision=8)
nPoints = 25  # n umber of integration points
upperBound = 30  # truncation of integral
# Gauss-Legendre
[uv, wgtv] = GL_integration(upperBound, nPoints)
uv = uv.reshape(nPoints, 1)
print(str(uv).replace(' [', '').replace('[', '').replace(']', ''))
wgtv = wgtv.reshape(nPoints, 1)
print(str(wgtv).replace(' [', '').replace('[', '').replace(']', ''))
# Trapezoid
[uv, wgtv] = trap_integration(0, upperBound, nPoints)
uv = uv.reshape(nPoints, 1)
print(str(uv).replace(' [', '').replace('[', '').replace(']', ''))
wgtv = wgtv.reshape(nPoints, 1)
print(str(wgtv).replace(' [', '').replace('[', '').replace(']', ''))
"""