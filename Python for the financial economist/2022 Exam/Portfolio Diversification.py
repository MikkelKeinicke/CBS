from pandas_datareader.famafrench import FamaFrenchReader
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import matplotlib.ticker as ticker
from visualization import DefaultStyle, default_colors
DefaultStyle()
import pandas as pd
import sys
np.set_printoptions(threshold = sys.maxsize)

#Import data
reader = FamaFrenchReader("49_Industry_Portfolios", start=datetime.datetime(1970, 1, 1))
industry_port = reader.read()
print(industry_port["DESCR"])

#Convert to decimal
ind_vw_return = industry_port[0] / 100

num_companies = industry_port[4]
avg_firm_size = industry_port[5]
sector_mktcap = num_companies * avg_firm_size
total_mkt_cap = sector_mktcap.sum(axis=1)   #axis 1 meaning to sum across columns (fixed row)

sector_weight = sector_mktcap.divide(total_mkt_cap, axis="rows")

#check
sector_total_weight = sector_weight.sum(axis="columns")
print(sector_weight)

#date = np.array(sector_weight.index[:,])
def date_range(start,end):
    delta = end - start
    days = [start + delta(months = i) for i in range(delta.months + 1)]
    return days

start_date = datetime.datetime(1970,1,1)
end_date = datetime.datetime(2022,10,1)
print(date_range(start_date,end_date))

asset_list = np.array(sector_weight.columns.astype('str'))
weights = np.array(sector_weight).astype(float)

print("date=", np.array(date))
print(asset_list)
print(weights)

#weights = np.array([[0.2,0.8],[0.4,0.6]])
#date = np.array(['2022-01','2022-02'])
#asset_list = np.array(['A','B'])

#Plot
fig, ax = plt.subplots(figsize=(16, 6))

ax.stackplot(np.array(date), weights.T, labels=asset_list)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=6)
ax.set_xlabel("Time")
ax.set_ylabel("Portfolio weights")
ax.set_title("Sector weights")
plt.show()