from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

##Question 1
t_values = np.linspace(-5, 5, 10000)
df1 = 2
df2 = 10
df3 = 50

tdist1 = stats.t.pdf(t_values, df1)
tdist2 = stats.t.pdf(t_values, df2)
tdist3 = stats.t.pdf(t_values, df3)

snorm = stats.norm.pdf(t_values)

plt.plot(t_values, tdist1, linestyle="--")
plt.plot(t_values, tdist2, linestyle="--")
plt.plot(t_values, tdist3, linestyle="--")
plt.plot(t_values, snorm)
plt.xlabel("t")
plt.ylabel("Density")
plt.legend(["$v$=2,", "$v$=5", "$v$=10", "N(0,1)"])
plt.show()

##Question 2
n = 10
mu = 5
sigma = 2
sims = 10000

draws = np.random.normal(mu, sigma,size=(sims, n))
sample_mean = np.mean(draws, axis=1)
sample_var = np.std(draws, axis=1)**2
z_score = (sample_mean-mu)/np.sqrt(sample_var/(n-1))

t_values = np.linspace(-5,5,10000)
tdist = stats.t.pdf(t_values, n-1)
dist = stats.norm.pdf(t_values)

plt.hist(z_score, bins=50, density=True)
plt.plot(t_values, dist)
plt.plot(t_values, tdist)
plt.show()

##Question 3
n = 25
mu = 5
sigma = 2
sims = 10000

draws = np.random.normal(mu, sigma,size=(sims, n))
sample_mean = np.mean(draws, axis=1)
sample_var = np.std(draws, axis=1)**2
z_score = (sample_mean-mu)/np.sqrt(sample_var/(n-1))

t_values = np.linspace(-5,5,10000)
tdist = stats.t.pdf(t_values, n-1)
dist = stats.norm.pdf(t_values)

plt.figure(figsize=(10,6))
plt.hist(z_score, bins=50, density=True, label="Simulated")
plt.plot(t_values, dist, color="black", linestyle="--", label="Standard Normal")
plt.plot(t_values, tdist, color="gray", linestyle="--", label="True t-dist")
plt.legend()
plt.grid()
plt.xlabel("$t$")
plt.ylabel("Density")
plt.show()

#Question 4
def g(t):
    return np.exp(5+10*t)

print(stats.t.expect(g, args=(5,)))
#inf == not defined

#Question 5
def h(t):
    return 5*t**2
print(stats.t.expect(h, args=(5,)))













