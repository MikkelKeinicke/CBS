#from Integration import GL_integration, trap_integration
import numpy as np

def GL_integration(xmax, N):
    # Computes integration points and weights for Gauss-Legendre scheme
    # integrates from 0 to xmax with N integration points
    [u, wgtv] = np.polynomial.legendre.leggauss(N)
    u = u + 1
    u = u * xmax / 2  # integration points
    wgtv = wgtv * xmax / 2  # integration weights
    return u, wgtv

def trap_integration(xmin, xmax, N):
    # Computes integration points and weights for trapezoid scheme
    # integrates from xmin to xmax with N integration points
    u = np.linspace(xmin, xmax, N)
    wgtv = np.ones(N)
    wgtv[[0, N - 1]] = 1 / 2
    wgtv = wgtv * (xmax - xmin) / (N - 1)
    return u, wgtv

