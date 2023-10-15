import numpy as np
import matplotlib.pyplot as plt

S = 1000
r = 0.02
delta = 0.01
sigma = 0.3
T = 3/12
K = 1100
s = np.log(S)

u = np.arange(10**-8, 100, 0.1)

#2.2
def CharFunc(u):

    psi = np.exp(1j * u * (s + (r-delta - 1/2 * sigma**2) * T) - 1/2 * u**2 * sigma**2 * T)

    return psi

Integrand2 = np.real(np.exp(-1j * u * np.log(K)) * CharFunc(u) / (1j * u))
Integrand1 = np.real(np.exp(-1j * u * np.log(K)) * CharFunc(u - 1j) / (1j * u * S * np.exp((r-delta)*T)))

plt.plot(u, Integrand1, u, Integrand2)
plt.legend(["Integrand 1", "Integrand 2"])
plt.xlabel("u")
plt.show()

#2.3
def CallPrice(u, w):

    integrand1 = np.real(np.exp(-1j * u * np.log(K)) * CharFunc(u - 1j) / (1j * u * S * np.exp((r - delta) * T)))
    integral1= np.sum(w * integrand1)
    pi1 = 1/2 + 1/np.pi * integral1

    integrand2 = np.real(np.exp(-1j * u * np.log(K)) * CharFunc(u) / (1j * u))
    integral2 = np.sum(w * integrand2)
    pi2 = 1/2 + 1/np.pi * integral2

    return S * np.exp(-delta*T) * pi1 - K * np.exp(-r * T) * pi2

#GL
u = np.array([0.65233679, 3.37341583, 8.01476079, 14.16511515, 21.27814153, 28.72185847, 35.83488485, 41.98523921, 46.62658417, 49.34766321])
w = np.array([1.66678361, 3.73628373, 5.47715906, 6.73166798, 7.38810562, 7.38810562, 6.73166798, 5.47715906, 3.73628373, 1.66678361])
C_GL = CallPrice(u, w)

#Trap
u = np.array([0.00000001, 5.55555556, 11.1111111, 16.6666667, 22.2222222, 27.7777778, 33.3333333, 38.8888889, 44.4444444, 50])
w = np.array([2.77777778, 5.55555555, 5.55555555, 5.55555555, 5.55555555, 5.55555555, 5.55555555, 5.55555555, 5.55555555, 2.77777778])
C_Trap = CallPrice(u, w)

print(C_GL, C_Trap)
