import pandas as pd
import numpy as np

#læs CSV fil
csv = pd.read_csv("C:\\Users\\Mikkel\\Desktop\\CSV.csv")
print(csv)

#slet en række
csv = csv.drop([1152])
print(csv)

#sum kolonner
sum = csv.apply(np.sum)
print(sum)

#eksporter til CSV
sum.to_csv("C:\\Users\\Mikkel\\Desktop\\CSV2.csv")
