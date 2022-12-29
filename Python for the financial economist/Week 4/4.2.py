import numpy as np
import datetime
from pandas_datareader import DataReader
import matplotlib.pyplot as plt

#Question 1
def drawdown(data: np.ndarray):

    datamax = data.cummax()
    drawdowns = (data - datamax)/datamax

    return drawdowns, datamax

def maxdrawdown(data: np.ndarray):

    return drawdown(data)[0].min()

#Question 2
sp500_adjclose = DataReader("^GSPC", "yahoo", start=datetime.date(1990, 1, 1))["Adj Close"]
plt.plot(sp500_adjclose)
plt.grid()
plt.show()

drawdownSP = drawdown(data = sp500_adjclose)[0]
plt.plot(drawdownSP)
plt.grid()
plt.show()

maxdrawdown = maxdrawdown(data = sp500_adjclose)
print(maxdrawdown)