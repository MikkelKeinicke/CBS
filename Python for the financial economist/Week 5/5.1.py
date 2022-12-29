from scipy import stats
from scipy import optimize

#Question 1
func = lambda x: -x[0]**2 - x[1]**2 - x[2]**2

res = optimize.minimize(fun = lambda x: -func(x), x0 = [1,1,1])   #We need to maximize, so the function is negative func
print(res)  #solution is when they are 0

#Question 2
Hessian = [[-2, 0, 0],
           [0, -2, 0],
           [0, 0, -2]]


