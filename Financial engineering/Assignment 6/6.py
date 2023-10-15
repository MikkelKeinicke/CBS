import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import norm
np.set_printoptions(precision=4)

S = 1000
r = 0.02
delta = 0.01
sigma = 0.3
K = 1000

dt = 1/252
T = 1/12
steps = int(T/dt)

#alternative simulation
"""
def GBM_sim(paths, steps):
    size = (steps+1)
    prices = np.zeros(size)
    prices[0] = S

    eps = np.random.normal(size = steps)

    for t in range(steps):
        prices[t+1] = prices[t] * np.exp((r - delta - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * eps[t])
    return prices[1:]

for k in range(len(AC)):
    sim = GBM_sim(paths, steps)
    AC[k] = np.maximum(0, np.mean(sim) - K) * np.exp(-r * T)
AC = np.mean(AC)
print(AC)
"""

#simulate
def GBM_sim(paths, steps):

    size = (paths, steps+1)
    prices = np.zeros(size)
    prices[:,0] = S
    eps = np.random.normal(size = (paths, steps))

    for t in range(steps):
        prices[:, t+1] = prices[:,t] * np.exp((r - delta - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * eps[:,t])
    return prices[:,1:]

#plot
#x = np.arange(0, T+dt, dt)
#plt.plot(x, Y.T)
#plt.show()

#Asian call
np.random.seed(1)
paths = 1000
S_sim = GBM_sim(paths, steps)
AC = np.zeros(paths)

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)
AC_P = np.mean(AC)
CI = st.norm.interval(confidence=0.95, loc=AC_P, scale = st.sem(AC))
print("Asian Call option price = ", AC_P, "\n95% confidence interval =", CI, "\n")

######6.2######
#Antithetic variates
def GBM_sim_AV(paths, steps):

    size = (2 * paths, steps+1)
    prices = np.zeros(size)
    prices[:,0] = S
    eps = np.random.normal(size = (paths, steps))
    eps = np.vstack((eps, -eps))

    for t in range(steps):
        prices[:, t+1] = prices[:,t] * np.exp((r - delta - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * eps[:,t])
    return prices[:,1:]

#Asian call
np.random.seed(1)
paths = 500
S_sim = GBM_sim_AV(paths, steps)
AC = np.zeros(2 * paths)

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)

AC_avg = (AC[0:500] + AC[500:])/2
AC_P = np.mean(AC)
CI = st.norm.interval(confidence=0.95, loc=AC_P, scale = st.sem(AC_avg))
print("Asian Call option price with antithetic variates = ", AC_P, "\n95% confidence interval", CI)

######6.3######
###### Terminal price ######
np.random.seed(1)
paths = 1000
S_sim = GBM_sim(paths, steps)
AC = np.zeros(paths)
EX = S * np.exp((r-delta)*T)    #we can compute E[X] analytically

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)

S_T = S_sim[:,-1]
correlation = np.corrcoef(AC, S_T)[0][1]    #[0][1] gives the correlation in the 2x2 matrix
b_star = np.cov(AC, S_T)[0][1]/np.var(S_T)
AC_b = AC - b_star * (S_T-EX)

CI = st.norm.interval(confidence=0.95, loc=np.mean(AC_b), scale = st.sem(AC_b))
print("\nAsian Call option, control variate: terminal stock price = ", np.mean(AC_b), "\n95% confidence interval", CI)

#save subplot for total plot in the end
plt.subplot(221)
plt.scatter(S_T,AC)
plt.title('S_T')

###### Payoff regular call option ######
def BSM_C(S, K, r, delta, sigma, T):
    d1 = (np.log(S/K) + (r - delta + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * np.exp(-delta*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return C

np.random.seed(1)
S_sim = GBM_sim(paths, steps)
AC = np.zeros(paths)
EX = np.exp(r*T) * BSM_C(S, K, r, delta, sigma, T)    #we can compute E[X] analytically with BSM

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)

X = np.maximum(0, S_sim[:,-1] - K)
correlation = np.corrcoef(AC, X)[0][1]    #[0][1] gives the correlation in the 2x2 matrix
b_star = np.cov(AC, X)[0][1]/np.var(X)
AC_b = AC - b_star * (X-EX)

CI = st.norm.interval(confidence=0.95, loc=np.mean(AC_b), scale = st.sem(AC_b))
print("\nAsian Call option, control variate: regular call option = ", np.mean(AC_b), "\n95% confidence interval", CI)

#save subplot for total plot in the end
plt.subplot(222)
plt.scatter(X,AC)
plt.title('Call option')


###### Arithmetic average ######
np.random.seed(1)
S_sim = GBM_sim(paths, steps)
AC = np.zeros(paths)
X= np.zeros(paths)
EX = 1/steps*S*(np.exp((r-delta)*dt)-np.exp((r-delta)*dt*(steps+1)))/(1-np.exp((r-delta)*dt))    #Closed-form expression

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)
    X[k] = np.mean(S_sim[k,:])

correlation = np.corrcoef(AC, X)[0][1]    #[0][1] gives the correlation in the 2x2 matrix
b_star = np.cov(AC, X)[0][1]/np.var(X)
AC_b = AC - b_star * (X-EX)

CI = st.norm.interval(confidence=0.95, loc=np.mean(AC_b), scale = st.sem(AC_b))
print("\nAsian Call option, control variate: arithmetic average = ", np.mean(AC_b), "\n95% confidence interval", CI)

#save subplot for total plot in the end
plt.subplot(223)
plt.scatter(X,AC)
plt.title('Arithmetic average')

###### Geometric average ######
np.random.seed(1)
S_sim = GBM_sim(paths, steps)
AC = np.zeros(paths)
X= np.zeros(paths)

i = np.arange(1, steps+1)
T_bar = 1 / steps * np.sum(i * dt)
sigma_bar = np.sqrt(sigma**2 / (steps**2 * T_bar) * np.sum((2 * i - 1) * (steps + 1 - i) * dt))
delta_bar = delta + 1/2 * sigma**2 - 1/2 *sigma_bar**2

EX = np.exp(r*T_bar) * BSM_C(S, K, r, delta_bar, sigma_bar, T_bar)

for k in range(len(AC)):
    AC[k] = np.maximum(0, np.mean(S_sim[k,:]) - K) * np.exp(-r * T)
    X[k] = np.maximum(0, np.prod(S_sim[k,:])**(1/steps) - K)

correlation = np.corrcoef(AC, X)[0][1]    #[0][1] gives the correlation in the 2x2 matrix
b_star = np.cov(AC, X)[0][1]/np.var(X)
AC_b = AC - b_star * (X-EX)

CI = st.norm.interval(confidence=0.95, loc=np.mean(AC_b), scale = st.sem(AC_b))
print("\nAsian Call option, control variate: geometric average = ", np.mean(AC_b), "\n95% confidence interval", CI)

#save subplot for total plot in the end
plt.subplot(224)
plt.scatter(X,AC)
plt.title('Geomtric average')
plt.show()