import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader.famafrench import FamaFrenchReader, get_available_datasets
import datetime

from pandas_datareader.data import DataReader
from pandas_datareader.famafrench import FamaFrenchReader, get_available_datasets

#Problem 1
#Question 1
#csv = pd.read_csv("5_Industry_Portfolios.csv")
reader = FamaFrenchReader("5_Industry_Portfolios", start=datetime.datetime(1990, 1, 1))
industry_port_daily = reader.read()

ind_eq_weighted = industry_port_daily[1]

ind_mc_weighted = industry_port_daily[0]

print(ind_eq_weighted)

#Question 2
eq_log_returns = np.log1p(ind_eq_weighted/100.0)
print(eq_log_returns.head())

#Question 3
mu = np.mean(eq_log_returns.values, axis=0)
cov_mat = np.cov(eq_log_returns.values.T)
print(mu)
print(cov_mat)

#Question 4
mu_1 = mu*12
mu_5 = mu*60

cov_mat_1 = cov_mat*12
cov_mat_5 = cov_mat*60

#Question 5
lmbd = 10
B1 = np.ones_like(mu_1)
B5 = np.ones_like(mu_5)
A1 = np.transpose(B1)
A5 = np.transpose(B5)
C1 = A1 @ np.linalg.inv(cov_mat_1) @ np.transpose(A1)
C1inv = 1 / C1
C5 = A5 @ np.linalg.inv(cov_mat_5) @ np.transpose(A5)
C5inv = 1/C5

w_1 = 1/lmbd * np.linalg.inv(cov_mat_1) @ mu_1 - 1/lmbd * np.linalg.inv(cov_mat_1) @ np.transpose(A1) * C1inv * (A1 @ np.linalg.inv(cov_mat_1) @ mu_1 - lmbd*B1)
w_5 = 1/lmbd * np.linalg.inv(cov_mat_5) @ mu_5 - 1/lmbd * np.linalg.inv(cov_mat_5) @ np.transpose(A5) * C5inv * (A5 @ np.linalg.inv(cov_mat_5) @ mu_5 - lmbd*B5)
print(w_1)
print(w_5)

#Question 6
"We cannot use log-returns and log-variance to calculate sumproducts against weights"

#Question 7

#Question 8