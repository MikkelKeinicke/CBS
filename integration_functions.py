import numpy as np

def GL_integration(xmax, N):
    # Computes integration points and weights for Gauss-Legendre scheme
    # integrates from 0 to xmax with N integration points
    [uv, wgtv] = np.polynomial.legendre.leggauss(N)
    uv = uv + 1
    uv = uv * xmax / 2  # integration points
    wgtv = wgtv * xmax / 2  # integration weights
    return uv, wgtv

def trap_integration(xmin, xmax, N):
    # Computes integration points and weights for trapezoid scheme
    # integrates from xmin to xmax with N integration points
    uv = np.linspace(xmin, xmax, N)
    wgtv = np.ones(N)
    wgtv[[0, N - 1]] = 1 / 2
    wgtv = wgtv * (xmax - xmin) / (N - 1)
    return uv, wgtv