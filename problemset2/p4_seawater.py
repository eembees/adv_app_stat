import numpy as np
import pandas as pd
import lib.lib_math as lm
from lib.monte_carlo import monte_carlo
from lib.plot_gaussian import plot_gaussian
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from pathlib import Path


plt.xkcd()


# custom functions from lib modified
def llh(data, pdf):
    try:
        loglh = (np.sum(np.log(pdf(data)), axis=0))
    except ValueError:
        loglh = (np.sum([np.log(pdf(dat)) for dat in data]))
    return loglh


# import data
arr97 = np.loadtxt('GlobalTemp_1.txt', skiprows=1, )[6]
arr17 = np.loadtxt('GlobalTemp_2.txt', skiprows=1, )[6]

# filter data
# arr97 = np.ma.masked_where(arr97 == -99.99, arr97)
# arr17 = np.ma.masked_where(arr17 == -99.99, arr17)

arr97 = arr97[arr97 != -99.99]
arr17 = arr17[arr17 != -99.99]

# make sample data for plotting
x_arr = np.linspace(start=-2, stop=4, num=100)

# initialize figure
figs, axs = plt.subplots(figsize=(7.5, 4))

# # Make a epan. kernel bw 0.4


# 1997 KDE homemade
p_KDE_97 = lm.make_KDE(arr97, type='epan', h=0.4)
y_97 = [p_KDE_97(x_y) for x_y in x_arr]

integral_97_big = quad(p_KDE_97, -2, 4)
integral_97_small = quad(p_KDE_97, -2, 0)

label97 = '1997 KDE:  \n$\int_{{-2}}^{{+4}} P = {0:.2f},$\n$\int_{{-2}}^{{0}} P = {1:.2f}$'.format(integral_97_big[0],
                                                                                                   integral_97_small[0])

axs.plot(x_arr, y_97, linestyle='dashed', label=label97, alpha=0.7)

# 2017 KDE homemade
p_KDE_17 = lm.make_KDE(arr17, type='epan', h=0.4)
y_17 = [p_KDE_17(x_y) for x_y in x_arr]

integral_17_big = quad(p_KDE_17, -2, 4)
integral_17_small = quad(p_KDE_17, -2, 0)

label17 = '2017 KDE: \n $\int_{{-2}}^{{+4}} P = {0:.2f},$\n$\int_{{-2}}^{{0}} P = {1:.2f}$'.format(integral_17_big[0],
                                                                                                   integral_17_small[0])

axs.plot(x_arr, y_17, linestyle='dashed', label=label17)
'''
# 1997 KDE sklearn

p_KDE_97_sk = KernelDensity(kernel='epanechnikov',
                            bandwidth=0.4).fit(arr97.reshape(-1, 1))
y_97_sk = np.exp(p_KDE_97_sk.score_samples(x_arr.reshape(-1, 1)))


def integration_function_97(x):
    return np.exp(p_KDE_97_sk.score_samples(np.array(x).reshape(-1, 1)))


integral_97_big_sk = quad(integration_function_97, -2, 4)
integral_97_small_sk = quad(integration_function_97, -2, 0)

label97_sk = '1997 KDE (sklearn):  \n$\int_{{-2}}^{{+4}} P = {0:.2f},$\n$\int_{{-2}}^{{0}} P = {1:.2f}$'.format(
    integral_97_big_sk[0], integral_97_small_sk[0])

axs.plot(x_arr, y_97_sk, linestyle='dotted', color='xkcd:hot pink', label=label97_sk, alpha=0.5)

# 2017 KDE sklearn
p_KDE_17_sk = KernelDensity(kernel='epanechnikov',
                            bandwidth=0.4).fit(arr17.reshape(-1, 1))
y_17_sk = np.exp(p_KDE_17_sk.score_samples(x_arr.reshape(-1, 1)))


def integration_function_17(x):
    return np.exp(p_KDE_17_sk.score_samples(np.array(x).reshape(-1, 1)))


integral_17_big_sk = quad(integration_function_17, -2, 4)
integral_17_small_sk = quad(integration_function_17, -2, 0)

label17_sk = '2017 KDE (sklearn):  \n$\int_{{-2}}^{{+4}} P = {0:.2f},$\n$\int_{{-2}}^{{0}} P = {1:.2f}$'.format(
    integral_17_big_sk[0], integral_17_small_sk[0])

axs.plot(x_arr, y_17_sk, linestyle='dotted', color='xkcd:lavender', label=label17_sk, alpha=0.5)
'''
# generate 1000 MC points from 1997, and find llh ratio (1997/2017)

num_mc = 100
mc_ratios = []
for i in range(num_mc):
    mc_samples = monte_carlo(p_KDE_97, range_x=(-1, 2))

    # mc_samples = monte_carlo(integration_function_97,range_x=(-1,2))

    llh_97 = llh(mc_samples, pdf=p_KDE_97)
    llh_17 = llh(mc_samples, pdf=p_KDE_17)

    # lh_97 = np.exp(llh_97)
    # lh_17 = np.exp(llh_17)

    lh_rat = llh_97 - llh_17

    # loglh_diff = llh_97 - llh_17
    # lamd = -2 * loglh_diff
    # # print(lh_rat)
    mc_ratios.append(lh_rat)
    # TODO check on LLH ratios

f_out_name = Path.cwd() / 'sletfjerding_KDE_1000_samples.txt'

np.savetxt(fname=f_out_name, X=mc_samples)
label_hist = '1000 MC samples'

# # Make a histogram
hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(mc_samples, range_x=(-1, 2), N_bins=50)

axs.errorbar(hist_x[hist_mask], hist_y[hist_mask], yerr=hist_sy[hist_mask], label=label_hist,
             fmt='+', ecolor='xkcd:lavender', elinewidth=0.5)

# include distribution of mc results
ins = axs.inset_axes([-2, 0.4, 2, 0.5], transform=axs.transData)

ins = plot_gaussian(data=np.array(mc_ratios), ax=ins, nBins=15, short_text=True)
ins.set_title('Distribution of {} $\lambda$s'.format(len(mc_ratios)))

axs.set_title('Sea Surface Water Temperatures')
axs.set_xlabel('Temperature (C)')
axs.set_ylabel('$P_{KDE}$')
axs.set_ylim(0, 1)

legend = axs.legend(loc='upper right')

figs.tight_layout()
figs.savefig('p4_seawater_xkcd.pdf')
