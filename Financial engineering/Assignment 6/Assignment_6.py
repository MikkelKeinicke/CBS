from BSM_functions import BSM
import numpy as np
from numpy import sqrt, log, exp, mean, max
from numpy.random import default_rng
import scipy.stats as st
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

def simPathGBM():
    # Simulates path for GBM
    path = np.zeros(n+1)  # preallocating
    path[0] = S
    eps = np.random.normal(size = (n,1))
    for j in range(n):
        path[j+1] = path[j]*exp((r-delta-0.5*sigma**2)*dt + sigma*sqrt(dt)*eps[j])
    return path[1:]

def simPathGBM_AV():
    # Simulates path for GBM
    path = np.zeros(n+1)  # preallocating
    path[0] = S
    for j in range(n):
        path[j+1] = path[j]*exp((r-delta-0.5*sigma**2)*dt + sigma*sqrt(dt)*eps[j])
    return path[1:]

#### Solution to assignment
S = 1000
s = log(S)
r = 0.02
delta = 0.01
sigma = 0.30
dt = 1/252

K = 1000
T = 1/12
n = int(T/dt)

nSim = 1000
estimate = np.zeros(6)
CI = np.zeros((6,2))
correlation = np.zeros(4)

## Regular MC
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
estimate[0] = np.mean(Y)
print(Y.shape)
CI[0,:] = st.norm.interval(confidence=0.95,loc=np.mean(Y),scale=st.sem(Y))
print(estimate[0], CI[0,:])


## Antithetic variates
np.random.seed(1) #resetting random number generator
nSim_AV = int(nSim/2)
Y = np.zeros(nSim_AV)
Ytilde = np.zeros(nSim_AV)
for j in range(nSim_AV):
    # original path
    eps = np.random.normal(size=(n, 1)) # original eps
    path = simPathGBM_AV()
    Y[j] = exp(-r * T) * max([0, mean(path) - K])
    # antithetic path
    eps = -eps # flipping sign on already generated eps
    path = simPathGBM_AV()
    Ytilde[j] = exp(-r * T) * max([0, mean(path) - K])
Y_ave = (Y+Ytilde)/2
estimate[1] = np.mean(Y_ave)
CI[1,:] = st.norm.interval(confidence=0.95,loc=np.mean(Y_ave),scale=st.sem(Y_ave))
print(estimate[1], CI[1,:])

## Control variates with terminal value
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = S*exp((r-delta)*T)
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   S_T = path[-1]
   X[j] = S_T
correlation[0] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
estimate[2] = np.mean(Yb)
CI[2,:] = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
print(estimate[2], CI[2,:])

plt.subplot(221)
plt.scatter(X,Y)
plt.title('S_T')
plt.show()


## Control variates with standard call price
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = exp(r*T)*BSM(S, K, delta, r, sigma, T, "call")
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   S_T = path[-1]
   X[j] = max([0,S_T-K])
correlation[1] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
estimate[3] = np.mean(Yb)
CI[3,:] = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
plt.subplot(222)
plt.scatter(X,Y)
plt.title('Call on S_T')
print(estimate[3], CI[3,:])

## Control variates with arithmetic average
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = 1/n*S*(exp((r-delta)*dt)-exp((r-delta)*dt*(n+1)))/(1-exp((r-delta)*dt))
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   X[j] = mean(path)
correlation[2] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
estimate[4] = np.mean(Yb)
CI[4,:] = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
plt.subplot(223)
plt.scatter(X,Y)
plt.title('Arithmetic average')
print(estimate[4], CI[4,:])

## Control variates with call on geometric average
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
iv = np.arange(1, n+1) #i running from 1 to n
T_bar = 1/n*sum(iv*dt)
sigma_bar = sqrt(sigma**2/(n**2*T_bar)*sum( (2*iv-1)*(n+1-iv)*dt ))
delta_bar = delta+1/2*sigma**2-1/2*sigma_bar**2
EX = exp(r*T_bar)*BSM(S, K, delta_bar, r, sigma_bar, T_bar, "call")
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   X[j] = max([0, np.prod(path)**(1/n) - K])
correlation[3] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
estimate[5] = np.mean(Yb)
CI[5,:] = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
plt.subplot(224)
plt.scatter(X,Y)
plt.title('Call on geometric average')
plt.show()
print(estimate[5], CI[5,:])
"""
print('Estimate:\n',estimate)
print('Width of confidence intervals:\n',CI[:,1]-CI[:,0])
print('Correlations:\n',correlation)

input()


#### For lecture, showing how the efficacy of antithetic variates
# and control variates (with arithmetic average) depends on moneyness
nSim = 1000
CI_width = np.zeros((3,3))
correlation = np.zeros(3)

#### ITM option
K = 900

## Regular MC
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y),scale=st.sem(Y))
CI_width[0,0] = CI[1]-CI[0]

## Antithetic variates
np.random.seed(1) #resetting random number generator
nSim_AV = int(nSim/2)
Y = np.zeros(nSim_AV)
Ytilde = np.zeros(nSim_AV)
for j in range(nSim_AV):
    # original path
    eps = np.random.normal(size=(n, 1)) # original eps
    path = simPathGBM_AV()
    Y[j] = exp(-r * T) * max([0, mean(path) - K])
    # antithetic path
    eps = -eps # flipping sign on already generated eps
    path = simPathGBM_AV()
    Ytilde[j] = exp(-r * T) * max([0, mean(path) - K])
Y_ave = (Y+Ytilde)/2
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y_ave),scale=st.sem(Y_ave))
CI_width[1,0] = CI[1]-CI[0]

## Control variates with arithmetic average
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = 1/n*S*(exp((r-delta)*dt)-exp((r-delta)*dt*(n+1)))/(1-exp((r-delta)*dt))
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   X[j] = mean(path)
correlation[0] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
CI = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
CI_width[2,0] = CI[1]-CI[0]
plt.subplot(131)
plt.scatter(X,Y)
plt.title('ITM')


#### ATM option
K = 1000

## Regular MC
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y),scale=st.sem(Y))
CI_width[0,1] = CI[1]-CI[0]

## Antithetic variates
np.random.seed(1) #resetting random number generator
nSim_AV = int(nSim/2)
Y = np.zeros(nSim_AV)
Ytilde = np.zeros(nSim_AV)
for j in range(nSim_AV):
    # original path
    eps = np.random.normal(size=(n, 1)) # original eps
    path = simPathGBM_AV()
    Y[j] = exp(-r * T) * max([0, mean(path) - K])
    # antithetic path
    eps = -eps # flipping sign on already generated eps
    path = simPathGBM_AV()
    Ytilde[j] = exp(-r * T) * max([0, mean(path) - K])
Y_ave = (Y+Ytilde)/2
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y_ave),scale=st.sem(Y_ave))
CI_width[1,1] = CI[1]-CI[0]

## Control variates with arithmetic average
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = 1/n*S*(exp((r-delta)*dt)-exp((r-delta)*dt*(n+1)))/(1-exp((r-delta)*dt))
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   X[j] = mean(path)
correlation[1] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
CI = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
CI_width[2,1] = CI[1]-CI[0]
plt.subplot(132)
plt.scatter(X,Y)
plt.title('ATM')


#### OTM option
K = 1100

## Regular MC
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y),scale=st.sem(Y))
CI_width[0,2] = CI[1]-CI[0]

## Antithetic variates
np.random.seed(1) #resetting random number generator
nSim_AV = int(nSim/2)
Y = np.zeros(nSim_AV)
Ytilde = np.zeros(nSim_AV)
for j in range(nSim_AV):
    # original path
    eps = np.random.normal(size=(n, 1)) # original eps
    path = simPathGBM_AV()
    Y[j] = exp(-r * T) * max([0, mean(path) - K])
    # antithetic path
    eps = -eps # flipping sign on already generated eps
    path = simPathGBM_AV()
    Ytilde[j] = exp(-r * T) * max([0, mean(path) - K])
Y_ave = (Y+Ytilde)/2
CI = st.norm.interval(confidence=0.95,loc=np.mean(Y_ave),scale=st.sem(Y_ave))
CI_width[1,2] = CI[1]-CI[0]

## Control variates with arithmetic average
np.random.seed(1) #resetting random number generator
Y = np.zeros(nSim)
X = np.zeros(nSim)
EX = 1/n*S*(exp((r-delta)*dt)-exp((r-delta)*dt*(n+1)))/(1-exp((r-delta)*dt))
for j in range(nSim):
   path = simPathGBM()
   Y[j] = exp(-r*T)*max([0, mean(path) - K])
   X[j] = mean(path)
correlation[2] = np.corrcoef(Y, X)[0][1]
b_star = np.cov(Y, X)[0][1]/np.var(X)
Yb = Y-b_star*(X-EX)
CI = st.norm.interval(confidence=0.95,loc=np.mean(Yb),scale=st.sem(Yb))
CI_width[2,2] = CI[1]-CI[0]
plt.subplot(133)
plt.scatter(X,Y)
plt.title('OTM')
plt.show()

print('Correlations:\n',correlation)
print('Width of confidence intervals:\n',CI_width)
"""