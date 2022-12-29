import numpy as np
import datetime
from pandas_datareader.famafrench import FamaFrenchReader
import matplotlib.pyplot as plt

#Question 1
reader = FamaFrenchReader("5_Industry_Portfolios", start=datetime.datetime(1990, 1, 1))
industry_port_daily = reader.read()
ind_eq_weighted = industry_port_daily[1]
log_ret = np.log(1+ind_eq_weighted/100) #divide by 100 because it is not in %

#Question 2
def resample(data: np.ndarray, num_sim: int, num_per: int = 1):
    #number of assets
    n = len(data)

    #get index for selecting data
    idx = np.random.randint(n, size=(num_sim, num_per)) #give random int from uniform distribution [0,n]. Size = output shape. num_sim*num_per = amount of samples

    return data[idx,:]

port_w = np.repeat(1 / 5, 5)    #array with 0.2 5 times
num_sim = 10000
num_per = 1

sim_log_returns = resample(log_ret.values, num_sim, num_per)
sim_lin_returns = np.exp(sim_log_returns) - 1   #transform to normal ret
sim_port_returns = sim_lin_returns @ port_w     #equally weighted port

VaR = np.percentile(sim_port_returns, 5)
cVaR = np.mean(sim_port_returns[sim_port_returns <= VaR])

plt.hist(sim_port_returns, density = True, bins = 15)
plt.axvline(VaR, color="black")
plt.axvline(cVaR, color="black")
plt.show()

#Question 3





