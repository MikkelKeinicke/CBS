S = 110
K = 90
T = 5
sigma = 0.5
r = 0.05

from BSM import BSM
print(BSM.Call(S, K, T, sigma, r))
print(BSM.Put(S, K, T, sigma, r))


from Car import Car
x = Car("Tesla")
y = Car("Fiat")

x.remove_wheels(1)
y.remove_wheels(2)

list = [x,y]

for i in list:
    i.remove_wheels(2)

print(x)
print(y)


