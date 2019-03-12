import numpy as np

import matplotlib.pyplot as plt

# Generate data
sigma = 20
arr_pt_1 = np.random.normal(loc=100, scale=sigma, size=23)
arr_pt_2 = np.random.normal(loc=170, scale=sigma, size=50)
arr_pt_3 = np.random.normal(loc=70, scale=sigma, size=50)
arr_pt_4 = np.random.normal(loc=20, scale=sigma, size=100 - 23)

arrs = [
    arr_pt_1,
    arr_pt_2,
    arr_pt_3,
    arr_pt_4,
]

realarr = np.concatenate([
    np.ones(shape=23)*100,
    np.ones(shape=50)*170,
    np.ones(shape=50)*70,
    np.ones(shape=100-23)*20,
])


trace = np.concatenate(arrs)

realx = [73, 23, 123]

fig, ax = plt.subplots()


ax.plot(trace, label='data')
ax.plot(realarr, label='ideal data', color='xkcd:lavender', ls='--')



for xline in realx:
        ax.axvline(x = xline, color='xkcd:easter green',)


plt.savefig('trace.pdf')