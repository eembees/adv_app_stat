import numpy as np
import pandas as pd
import lib.lib_math as lm
from scipy.integrate import quad
import matplotlib.pyplot as plt

# import data
arr97 = np.loadtxt(
    'GlobalTemp_1.txt',
    skiprows = 1,
)[6]
arr17 = np.loadtxt(
    'GlobalTemp_2.txt',
    skiprows = 1,
)[6]

# filter data
arr97 = np.ma.masked_where(arr97 == -99.99, arr97)
arr17 = np.ma.masked_where(arr17 == -99.99, arr17)

# print(arr97)

# # Make a epan. kernel bw 0.4
# TODO verify bw with another setup
p_KDE_97 = lm.make_KDE(arr97, type = 'epan')
p_KDE_17 = lm.make_KDE(arr17, type = 'epan')


# plotting

# make data for plotting
x_arr = np.linspace(start=-2, stop=4, num = 100)

y_97 = [p_KDE_97(x_y) for x_y in x_arr]
y_17 = [p_KDE_17(x_y) for x_y in x_arr]


# calculate integral

integral_97_big = quad(p_KDE_97, -2,4)
integral_17_big = quad(p_KDE_17, -2,4)
integral_97_small = quad(p_KDE_97, -2,0)
integral_17_small = quad(p_KDE_17, -2,0)


# make label strings
label97 = '1997 KDE:  $\int_{{-2}}^{{+4}} P = {0:.2f},\int_{{-2}}^{{0}} P = {1:.2f}$'.format(integral_97_big[0], integral_97_small[0])
label17 = '2017 KDE:  $\int_{{-2}}^{{+4}} P = {0:.2f},\int_{{-2}}^{{0}} P = {1:.2f}$'.format(integral_17_big[0], integral_17_small[0])



# generate 1000 MC points from 1997, and find llh ratio (1997/2017)


# plot it
figs, axs = plt.subplots()

axs.plot(x_arr, y_97, linestyle='dashed', label=label97)
axs.plot(x_arr, y_17, linestyle='dashed', label=label17)

axs.set_title('Sea Surface Water Temperatures')
axs.set_xlabel('Temperature')
axs.set_ylabel('$P_{KDE}$')

axs.legend()

figs.savefig('p4_seawater.pdf')