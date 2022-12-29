from BSM_functions import BSM_IV
import numpy as np
from numpy import log
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

S = 286.68
delta = 0
tau = 0.2219


### Question 1: Inferring r from put-call parity
K = 280
C_bid = 34.90
P_bid = 27.00
r_bid = -log((S+P_bid-C_bid)/K)/tau
C_ask = 35.25
P_ask = 27.30
r_ask = -log((S+P_ask-C_ask)/K)/tau
res = np.array([r_bid, r_ask])
print(res)
r = 0.02
input()

### Question 2: Computing implied vol smile
df = pd.read_excel(r'/Users/anderstrolle/Dropbox (CBS)/Python/CBS FE/Lecture1/Assignment_1.xlsx', sheet_name='TSLA')
print(df)
N = len(df.strike)  # number of options
IV = np.zeros(N)  # preallocating
for n in range(N):  # one strike at a time
    K = df.strike[n]
    optionType = df.type[n]
    price = df.price[n]
    sigma0 = 0.50  # initial guess
    IV[n] = BSM_IV(S, K, delta, r, sigma0, tau, optionType, price, 0.00001, 1000, "N")

# Plotting smile
mness = log(df.strike / S)
plt.plot(mness, IV)
plt.xlabel('Moneyness')
plt.show()

