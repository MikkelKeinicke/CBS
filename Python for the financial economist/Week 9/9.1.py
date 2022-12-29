from pandas_datareader.famafrench import FamaFrenchReader
import datetime
import numpy as np
import matplotlib.pyplot as plt

#Question 1
reader = FamaFrenchReader("12_Industry_Portfolios", start=datetime.datetime(1999, 1, 1))
industry_port = reader.read()
print(industry_port["DESCR"])
#from above we get that:
#0 = average monthly, value-weighted return (for each sector)
#1 = average monthly, equal-weighted return (for each sector)
#4 = number of firms in each sector
#5 = average firm size in each sector

#Equal weighted
ind_eq_weighted = industry_port[1] / 100    #turn returns into decimal. Use [1] which is equal-weighted return
ind_eq_weighted.columns = ind_eq_weighted.columns.str.strip()
equal_weighted_return = (ind_eq_weighted / 12 ).sum(axis = "columns")

#Value-weighted
#get vw-return
ind_vw_return = industry_port[0] / 100

num_companies = industry_port[4]
avg_firm_size = industry_port[5]
sector_mktcap = num_companies * avg_firm_size
total_mkt_cap = sector_mktcap.sum(axis=1)   #axis 1 meaning to sum across columns (fixed row)

sector_weight = sector_mktcap.divide(total_mkt_cap, axis="rows")

#VW-return
vw_return = (sector_weight * ind_vw_return).sum(axis="columns")

#Create index
mkt_index = np.array((1 + vw_return).cumprod())
eq_index = (1+equal_weighted_return).cumprod()

#Create graph
plt.plot(mkt_index)
plt.plot(eq_index)
plt.show()