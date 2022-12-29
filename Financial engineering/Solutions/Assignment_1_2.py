from BSM_functions import BSM
import numpy as np
from numpy import sqrt, log, exp, pi, real
import matplotlib.pyplot as plt
from integration_functions import GL_integration, trap_integration
np.set_printoptions(precision=4)


### Question 2: Plotting integrands
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

uv = np.arange(0, 100, 0.1)  # points at which integrand is evaluated
lowerBound = 10 ** (-8) # note that we cannot evaluate integrand at 0
uv[0] = lowerBound

charFct1 = charFuncBSM(uv - 1j) / (S * exp((r - delta) * tau))  # psi in pi2
integrand1 = real(exp(-1j * uv * log(K)) * charFct1 / (1j * uv))  # integrand in pi1
charFct2 = charFuncBSM(uv)  # psi in pi2
integrand2 = real(exp(-1j * uv * log(K)) * charFct2 / (1j * uv))  # integrand in pi2

plt.plot(uv, integrand1, uv, integrand2)
plt.gca().legend(('integrand 1', 'integrand 2'))
plt.xlabel('u')
plt.show()
input()


### Question 3: Comparison of integration schemes with analytical price
def callPrice():
    charFct1 = charFuncBSM(uv - 1j) / (S * exp((r - delta) * tau))  # psi in pi2
    integrand1 = real(exp(-1j * uv * log(K)) * charFct1 / (1j * uv))  # integrand in pi1
    integral = sum(wgtv * integrand1)  # computing integral by numerical integration
    P1 = 1 / 2 + 1 / pi * integral
    charFct2 = charFuncBSM(uv)  # psi in pi2
    integrand2 = real(exp(-1j * uv * log(K)) * charFct2 / (1j * uv))  # integrand in pi2
    integral = sum(wgtv * integrand2)  # computing integral by numerical integration
    P2 = 1 / 2 + 1 / pi * integral
    price = S * exp(-delta * tau) * P1 - K * exp(-r * tau) * P2  # call price
    return price

nPoints = 10  # n umber of integration points
upperBound = 50  # truncation of integral

# Price via the Gauss-Legendre integration scheme
[uv, wgtv] = GL_integration(upperBound, nPoints)
C_GL = callPrice()

# Price via trapezoid scheme
[uv, wgtv] = trap_integration(lowerBound, upperBound, nPoints)
C_Trap = callPrice()

# True price
C_true = BSM(S, K, delta, r, sigma, tau, "call")

res = np.array([C_true, C_GL, C_Trap])
print(res)
input()


### Question 4: Convergence to true price
nPointsv = np.linspace(5, 10, 6, dtype='int')
upperBound = 50
Cv = np.zeros((len(nPointsv), 3))  # preallocating
Cv[:, 0] = C_true
for i in range(len(nPointsv)):
    nPoints = nPointsv[i]
    # Price via the Gauss-Legendre integration scheme
    [uv, wgtv] = GL_integration(upperBound, nPoints)
    Cv[i, 1] = callPrice()
    # Price via trapezoid scheme
    [uv, wgtv] = trap_integration(lowerBound, upperBound, nPoints)
    Cv[i, 2] = callPrice()

print(Cv)

plt.plot(nPointsv, Cv)
plt.gca().legend(('analytical', 'Gauss-Legendre integration', 'trapezoid integration'))
plt.xlabel('Number of integration points')
plt.show()


# ### Printing integration points for Excel
# np.set_printoptions(precision=8)
# nPoints = 10  # n umber of integration points
# upperBound = 50  # truncation of integral
# # Gauss-Legendre
# [uv, wgtv] = GL_integration(upperBound, nPoints)
# uv = uv.reshape(nPoints, 1)
# print(str(uv).replace(' [', '').replace('[', '').replace(']', ''))
# wgtv = wgtv.reshape(nPoints, 1)
# print(str(wgtv).replace(' [', '').replace('[', '').replace(']', ''))
# Trapezoid
# [uv, wgtv] = trap_integration(lowerBound, upperBound, nPoints)
# uv = uv.reshape(nPoints, 1)
# print(str(uv).replace(' [', '').replace('[', '').replace(']', ''))
# wgtv = wgtv.reshape(nPoints, 1)
# print(str(wgtv).replace(' [', '').replace('[', '').replace(']', ''))
