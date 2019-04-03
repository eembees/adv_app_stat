import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import lib.lib_math as lm

plt.xkcd()

# import data
fname_data = list(Path.cwd().glob('*llhratios.txt'))[0]
# fname_data = list(Path.cwd().glob('*llhratios_random.txt'))[0]

data = np.loadtxt(fname_data)

# TODO use log scale

# set chisq dof
dof = 1

# fid number of points over 2.706
countpt = sum(data > 2.706)

sfstuff = 1 - stats.chi2.cdf(2.706, dof)
print(sfstuff)


# Get a histogram made
nBins=25
range_x = (0,8)
hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(data, N_bins=nBins, range_x=range_x)


# Do KS testing here

xarr = np.linspace(*range_x, num=nBins)
xarr_smooth = np.linspace(*range_x, num=nBins*10)

chi2_y = stats.chi2(dof).pdf(hist_x)



barwidth = (range_x[1] - range_x[0]) / nBins

KS_arr = hist_y - chi2_y

hist_c = np.cumsum(hist_y)
hist_c = hist_c - hist_c.min()
hist_c = hist_c / hist_c.max()

chi2_c = np.cumsum(chi2_y)
chi2_c = chi2_c - chi2_c.min()
chi2_c = chi2_c / chi2_c.max()


pval_KS = stats.ks_2samp(hist_y, chi2_y)

print(pval_KS)

# make and plot a chi square distribution
fig, axes = plt.subplots(nrows = 2 , figsize=(6,10))
ax, ax2 = axes.ravel()



# df = len(data) - 1
#
ax.plot(xarr_smooth, stats.chi2(dof).pdf(xarr_smooth),
        linestyle=':',
        label='$\int_{{2.706}}^{{\infty}}\chi^2(df= {}) = {:.4f}$'.format(dof,
                                                                          sfstuff))  # df is 1 because only one free parameter difference between the two


label_hist = 'LLH Ratio values'
ax.bar(hist_x, hist_y, barwidth, label = 'Hist of values', alpha=0.5, color = 'xkcd:dark red')

# ax.errorbar(hist_x[hist_mask], hist_y[hist_mask],
#             # yerr=hist_sy[hist_mask],
#             label=label_hist,
#             color='xkcd:light red',
#             fmt='+', elinewidth=0)
# ax.plot(hist_x, hist_y, ls='steps', label = 'Hist of values', alpha=0.5, color = 'xkcd:dark red')

ax.axvline(2.706, label='{} out of 100 are above 2.706'.format(countpt), color='xkcd:hot pink', linestyle='--')

ax.set_title('100 pseudo experiments of llh values')
ax.set_xlabel('$ \Lambda$')
ax.set_ylabel('Frequency')
ax.legend()

# ax2.errorbar(hist_x[hist_mask], hist_y[hist_mask],
#             # yerr=hist_sy[hist_mask],
#             label=label_hist,
#             fmt='+', ecolor='xkcd:lavender', elinewidth=0)
#
# ax2.errorbar(hist_x, chi2_ys, fmt = '3', color='xkcd:hot pink',
#         label='$\int_{{2.706}}^{{\infty}}\chi^2(df= {}) = {:.4f}$'.format(dof,
#         sfstuff))  # df is 1 because only one free parameter difference between the two

# ax2.plot(hist_x, hist_y, ls='steps', label = 'Hist of values', alpha=0.5, color = 'xkcd:dark red')
# ax2.plot(hist_x, chi2_y, ls='steps', label = 'Chi2 Distribution', alpha=0.5, color = 'xkcd:dark blue')

ax2.plot(hist_x, hist_c, ls='steps', label='ECDF', alpha=0.5, color = 'xkcd:light red')
ax2.plot(hist_x, chi2_c, ls='steps', label = '$\int_{{0}}^{{x}}\chi^2(df= {})$'.format(dof,
                                                                          ), alpha=0.5, color = 'xkcd:light blue')
# ax2.plot(xarr_smooth, stats.chi2(dof).cdf(xarr_smooth),label = 'Chi2 Distribution (CS)',alpha=0.5, color = 'xkcd:light blue',)


ax2.set_title('KS test stat = {:.3f}, pval = {:.3f}'.format(*pval_KS))

ax2.legend(loc='lower right')

fig.tight_layout()
fig.savefig('p5_llhratios_xkcd.pdf')
# fig.savefig('p5_llhratios.pdf')
