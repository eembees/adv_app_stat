import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)

sample = np.random.random(size=100000)

fig, ax = plt.subplots()

ax.hist(sample)

plt.show()

#
