import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from Integration import GL_integration, trap_integration

def CharfuncBSM(u):
    char = exp(1j * u * (s + (r - delta - 1/2 * sigma**2) * tau) - 1/2 * u**2 * sigma**2 * tau) #remember to use s=log(S)
    return char

S = 1000
s = np.log(S)
r = 0.02
delta = 0.01
sigma = 0.3
k = 1100
tau = 3/12

u = np.arange(0, 100, 0.1) # points at which integrand is evaluated. arange gives values from a range: (0) to (100) evenly spaced with 0.1.
lowerBound = 10**(-8) #use instead of 0
u[0] = lowerBound

charFct1 = CharfuncBSM(u)   #psi_t (u) in the first integrand (pi_2)
integrand1 = np.real(exp(-1j * u * np.log(k)) * charFct1 / (1j*u))   #the integrand

charFct2 = CharfuncBSM(u-1j)    #psi_t (u) in the second integrand (pi_1)
integrand2 = np.real(exp(-1j * u * np.log(k)) * charFct2 / (1j * u * S * exp((r - delta) * tau)))    #the integrand

plt.plot(u, integrand1, u, integrand2)
plt.gca().legend(('integrand 1', 'integrand 2'))
plt.xlabel('u')
plt.show()


############ 2.3 ############
def callPrice():
    ##Pi_2
    charFct1 = CharfuncBSM(u)
    integrand1 = np.real(exp(-1j * u * np.log(k)) * charFct1 / (1j*u))
    integral1 = sum(wgtv * integrand1) #computing integral by numerical integration
    Pi2 = 1/2 + 1/np.pi * integral1

    ##Pi_1
    charFct2 = CharfuncBSM(u-1j)
    integrand2 = np.real(exp(-1j * u * np.log(k)) * charFct2 / (1j * u * S * exp((r - delta) * tau)))
    integral2 = sum(wgtv * integrand2)  # computing integral by numerical integration
    Pi1 = 1/2 + 1/np.pi * integral2

    price = S * np.exp(-delta * tau) * Pi1 - k * np.exp(-r * tau) * Pi2
    return price

nPoints = 10 #number of integration points
upperBound = 50 #truncation of integral

#Price via Gauss-Legendre integration
[u, wgtv] = GL_integration(upperBound, nPoints)
C_GL = callPrice()
print(C_GL)

#Price via trapezoid scheme
[u, wgtv] = trap_integration(lowerBound, upperBound, nPoints)
C_Trap = callPrice()
print(C_Trap)

#True price (BSM)
from BSM import BSM
print(BSM.Call(S, k, tau, delta, sigma, r))


