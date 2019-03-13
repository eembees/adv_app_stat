import numpy as np
import matplotlib.pyplot as plt
from pathlib import  Path
from scipy import stats
import lib.lib_math as lm


# import data
fname_data = list(Path.cwd().glob('*llhratios.txt'))[0]

data = np.abs(np.loadtxt(fname_data) )


# make and plot a chi square distribution
fig, ax = plt.subplots()



xarr = np.linspace(start=0, stop=data.max() +1, num = 100)

dfs = range(4)
for df in dfs:
    ax.plot(xarr, stats.chi2(df).pdf(xarr),
            # linestyle=':',
            label='$\chi^2$, DoF={}'.format(df) )



hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(data, N_bins = 100)

label_hist = 'LLH Ratio values'

ax.errorbar(hist_x[hist_mask], hist_y[hist_mask], yerr=hist_sy[hist_mask], label =label_hist,
             fmt = '+', ecolor='xkcd:lavender',elinewidth=0)

ax.legend()


plt.show()