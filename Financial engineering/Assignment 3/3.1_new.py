import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

sigma = 0.1
lambd = 2
mu_j = -0.05
sigma_j = 0.1
sigma_bs = 0.1796
S = 1000
r = 0.02
delta = 0.01
tau = 3/12

#probabilities
def prob(lambd, k, tau):
    p = (np.exp(-lambd * tau) * lambd ** k * tau ** k) / np.math.factorial(k)
    return p

Prob = np.zeros(4)
Prob[0] = prob(lambd,0, tau)
Prob[1] = prob(lambd,1, tau)
Prob[2] = prob(lambd,2, tau)
Prob[3] = 1- Prob[0] - Prob[1] - Prob[2]
print(Prob)

##2
def C_BS(r, K, tau, sigma):
    d1 = (np.log(S/K) + (r-delta+1/2*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    C = np.exp(-delta*tau)*S*norm.cdf(d1) - np.exp(-r*tau)*K*norm.cdf(d2)
    return C

def MertonJump(K, tau, truncation):

    mu_bar = np.exp(mu_j + 1/2 * sigma_j**2) - 1
    lamb_prime = lambd * (1 + mu_bar)

    Price = 0

    for k in range(truncation):
        r_k = r - mu_bar * lambd + k / tau * (mu_j + 1/2 * sigma_j ** 2)
        sigma_k = np.sqrt(sigma ** 2 + k / tau * sigma_j ** 2)
        C_bs = C_BS(r_k, K, tau, sigma_k)

        Price = Price + prob(lamb_prime, k, tau) * C_bs

    return Price

#define for use
mness = np.arange(-0.2, 0.21, 0.01)
K = np.exp(mness) * S
tau = np.array([1/12, 3/12, 6/12 ,12/12])
truncation = 10

N = len(tau)    #number of maturities
M = len(K)      #number of strikes

P = np.zeros([M, N])
IV = np.zeros_like(P)

for n in range(N):
    tauv = tau[n]

    for i in range(M):
        Strike = K[i]
        P[i,n] = MertonJump(Strike, tauv, truncation)

        #Implied vol
        sigma_0 = 0.4
        for l in range(1000):
            d1 = (np.log(S / Strike) + (r - delta + 1 / 2 * sigma_0 ** 2) * tauv) / (sigma_0 * np.sqrt(tauv))
            vega = np.exp(-delta * tauv) * S * norm.pdf(d1) * np.sqrt(tauv)
            diff = C_BS(r, Strike, tauv, sigma_0) - P[i,n]

            if abs(diff)<0.00001:
                break
            sigma_0 = sigma_0 - diff/vega

        IV[i,n] = sigma_0

sigma_const = np.sqrt(sigma**2+lambd*((np.exp(2*mu_j+2*sigma_j**2)-2*np.exp(mu_j+1/2*sigma_j**2)+1)))
sigma_const = np.ones_like(P[:,1]) * sigma_const

plt.plot(mness, IV, mness, sigma_const)
plt.xlabel("Moneyness")
plt.legend(["1 month", "3 months", "6 months", "1 year", "Black-Scholes"], loc="upper center")
plt.show()

#4
tau = 1/4
ret = np.arange(-0.5, 0.51, 0.01)   #integrand evaluated at
sigma_const = np.sqrt(sigma**2+lambd*((np.exp(2*mu_j+2*sigma_j**2)-2*np.exp(mu_j+1/2*sigma_j**2)+1)))
truncation = 10
mu_bar = np.exp(mu_j + 1/2 * sigma_j**2)-1

pdf_Merton = 0
for k in range(truncation):
    sigma_k = np.sqrt(sigma ** 2 + k / tau * sigma_j ** 2)
    r_k = r - mu_bar * lambd + k / tau * (mu_j + 1 / 2 * sigma_j ** 2)
    prob = ((np.exp(-lambd * tau) * lambd ** k * tau ** k) / np.math.factorial(k))
    pdf_Merton = pdf_Merton + prob * norm((r_k-delta-1/2*sigma_k**2)*tau,sigma_k*np.sqrt(tau)).pdf(ret)

pdf_constVol = norm.pdf(ret, (r-delta-1/2*sigma_const**2)*tau,sigma_const*np.sqrt(tau))
plt.plot(ret, pdf_Merton)
plt.plot(ret, pdf_constVol)
plt.legend(["Merton model", "constant vol"], loc="upper right")
plt.xlabel("log-return")
plt.show()
