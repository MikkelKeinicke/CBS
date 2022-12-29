from pandas_datareader.famafrench import FamaFrenchReader
import datetime
import numpy as np
import matplotlib.pyplot as plt

reader = FamaFrenchReader("12_Industry_Portfolios", start=datetime.datetime(1999, 1, 1))
industry_port = reader.read()

#Question 1
ind_vw_return = industry_port[0] / 100
log_ret = np.log(1 + ind_vw_return)

#mean
mean_log_ret = np.mean(log_ret)
print(mean_log_ret)
#covariance
cov_mat_log_ret = log_ret.cov()
print(cov_mat_log_ret)

#plt.matshow(cov_mat_log_ret)
#plt.show()
#use his solution plot

#Question 2
def simulate_asset_price(mu: np.ndarray, cov_mat: np.ndarray,
                         horizon: float, dt: float, num_sim: int,
                         transform_input: bool=False):

    if transform_input:
        mu = mu * dt
        cov_mat = cov_mat * dt

    num_assets = len(mu)
    num_periods = int(horizon/dt)

    asset_prices = np.ones((num_sim, 1 + num_periods, num_assets))  #=shape, i.e., 3 dimensional: steps, paths, industries

    log_ret = np.random.multivariate_normal(mu, cov_mat, size=(num_sim, num_periods))

    asset_prices[:,1:,:] = np.exp(np.cumsum(log_ret, axis=1))

    return asset_prices

#Simulation:
num_sim = 5000
dt = 1
horizon = 5
mu = mean_log_ret * 12  #we can multiply by 12 because log-returns are additive
cov_mat = cov_mat_log_ret * 12

time_points = np.arange(0, horizon + 0.01, dt)  #used to graph

asset_prices = simulate_asset_price(mu = mu, cov_mat = cov_mat, horizon = horizon, dt = dt, num_sim = num_sim)

#buy-and-hold
port_w = np.ones(len(mu)) / len(mu)
buy_and_hold_index = asset_prices @ port_w

#check dimensions
print(asset_prices.shape, buy_and_hold_index.shape)

#define function for constant mix portfolio
def calculate_period_returns(index: np.ndarray):

    #the function essentially calculates period return as a non-lagged (t=t) divided by lagged (t=t-1)
    #-1 in the return is to go from gross return to return

    n = index.shape[1]

    #get matrix not lagged
    new_mat = index[:, 1:n]
    #get matrix lagged
    old_mat = index[:, 0:n-1]
    #get 1+return
    periodtr = new_mat / old_mat

    return periodtr - 1

def calculate_constant_mix_index(index: np.ndarray, weights: np.ndarray):

    port_index = np.ones((index.shape[0], index.shape[1]))

    #calculate period returns on assets
    per_ret = calculate_period_returns(index)

    #calculate port. period return
    port_per_ret = per_ret @ weights

    #calculate port. index
    port_index[:, 1:] = np.cumprod(1 + port_per_ret, axis=1)

    return port_index

constant_mix_index = calculate_constant_mix_index(asset_prices, port_w)

#Calculate percentiles
percentiles_buy_and_hold = np.percentile(buy_and_hold_index, [0.5, 1, 2.5, 5, 10, 50, 90, 95, 97.5, 99, 99.5],
                                         axis=0)
percentiles_constant_mix = np.percentile(constant_mix_index, [0.5, 1, 2.5, 5, 10, 50, 90, 95, 97.5, 99, 99.5],
                                         axis=0)

plt.plot(time_points, percentiles_buy_and_hold)
plt.plot(time_points, percentiles_constant_mix)
plt.show()




