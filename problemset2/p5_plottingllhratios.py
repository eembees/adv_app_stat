import numpy as np
import matplotlib.pyplot as plt
from pathlib import  Path
from scipy import stats
import lib.lib_math as lm


# import data
fname_data = list(Path.cwd().glob('*llhratios.txt'))[0]

data = np.abs(np.loadtxt(fname_data) )



# fid number of points over 2.706
countpt = sum(data > 2.706)


sfstuff = stats.chi2.sf(2.706, 2)
print(sfstuff)

# make and plot a chi square distribution
fig, ax = plt.subplots()



xarr = np.linspace(start=0, stop=data.max() +1, num = 100)

# dfs = range(4)
# for df in dfs:
#     ax.plot(xarr, stats.chi2(df).pdf(xarr),
#             # linestyle=':',
#             label='$\chi^2$, DoF={}'.format(df) )

# df = len(data) - 1

ax.plot(xarr, stats.chi2(1).pdf(xarr),
            linestyle=':',
            label='$\chi^2$, DoF={}'.format(1) ) # df is 1 because only one free parameter difference between the two





hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(data, N_bins = 100)

label_hist = 'LLH Ratio values'

ax.errorbar(hist_x[hist_mask], hist_y[hist_mask],
            # yerr=hist_sy[hist_mask],
            label =label_hist,
            fmt = '+', ecolor='xkcd:lavender',elinewidth=0)

ax.axvline(2.706, label = '{} out of 100 are above 2.706'.format(countpt))


ax.legend()

plt.show()