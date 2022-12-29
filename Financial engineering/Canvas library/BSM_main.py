from BSM_functions import BSM, BSM_IV

S = 100
K = 100
delta = 0.01
r = 0.02
sigma = 0.20
T = 1 / 4
optionType = "call"

price = BSM(S, K, delta, r, sigma, T, optionType)
print(price)
IV = BSM_IV(S, K, delta, r, sigma + 0.05, T, optionType, price, 0.00001, 1000, "Y")
print(IV)
