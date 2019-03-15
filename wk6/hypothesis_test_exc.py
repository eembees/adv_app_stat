import numpy as np
import lib.lib_math as lm
import matplotlib.pyplot as plt

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# Set parameters
nBins = 100
nRuns = 100000


mu_sig = 0
mu_bg  = 100


def llh_h0():
    pass

def llh_h1():
    pass


# generate data
data = np.random.poisson(lam=mu_bg / nBins, size = (nRuns, nBins))


# add a signal in some column
if mu_sig != 0:
    print('ooh')



lambdas = np.zeros(nRuns)

# print(data)

for i, data_row in enumerate(data):
    # calculate lambda
    nTot = data_row.sum()
    lam1 =  2 *    data_row[0]       * np.log((nBins/nTot) * data_row[0])
    lam2 =  2 * (nTot - data_row[0]) * np.log((nBins/nTot) * (nTot-data_row[0])/(nBins-1))

    lam1 = np.nan_to_num(lam1)

    lam = lam1 + lam2

    # append the sum of the data row to the lambdas (one experiment)
    lambdas[i] = lam#np.nansum(lam)



# make a histogram of the lam values
hist_x, hist_y, hist_sy, hist_mask = lm.bin_data( x=lambdas, N_bins=100 )
fig, ax = plt.subplots()

ax.errorbar(hist_x[hist_mask], hist_y[hist_mask],
            # yerr=hist_sy[hist_mask],
            label ='Histogram of $\lambda$ values',
            fmt = '+', ecolor='xkcd:lavender',elinewidth=0)


# add a chisq sdistribution
df = 1
xarr = np.linspace(hist_x.min(), hist_x.max(), num = 1000)
yarr_chi2 = stats.chi2(df).pdf(xarr)

ax.plot(xarr, yarr_chi2,
        linestyle=':',
        label='$\chi^2$, DoF={}'.format(df))  # df is 1 because only one free parameter difference between the two

ax.set_yscale('log')

ax.legend()


# plt.hist(lambdas)
plt.show()
