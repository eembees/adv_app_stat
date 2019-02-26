# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T10:44:16+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: plot_gaussian.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T10:44:50+01:00



from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def nice_string_output(names, values, extra_spacing = 0,):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]
##################################
##################################
#########MAIN FUNCTION############
##################################
##################################

def plot_gaussian(data, ax, nBins=100, textpos='l', legend=False):
    ### FITTING WITH A GAUSSIAN

    def func_gauss(x, N, mu, sigma):
        return N * stats.norm.pdf(x,mu,sigma)

    counts, bin_edges = np.histogram(data, bins=nBins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    s_counts = np.sqrt(counts)

    x = bin_centers[counts>0]
    y = counts[counts>0]
    sy = s_counts[counts>0]


    popt_gauss, pcov_gauss = curve_fit(func_gauss,x,y,p0=[1,data.mean(), data.std()])


    y_func = func_gauss(x, *popt_gauss)

    pKS = stats.ks_2samp(y, y_func)
    pKS_g1, pKS_g2 = pKS[0], pKS[1]

    # print('LOOK! \n \n \n pKS is {} \n \n \n '.format(pKS_g2))
    chi2_gauss  = sum((y - y_func)**2/ sy**2 )
    NDOF_gauss  = nBins - 3
    prob_gauss  = stats.chi2.sf(chi2_gauss, NDOF_gauss)


    namesl  = ['Gauss_N','Gauss_Mu','Gauss_Sigma',
                'KS stat', 'KS_pval',
                'Chi2 / NDOF','Prob']
    valuesl = ['{:.3f} +/- {:.3f}'.format(val, unc) for val, unc in zip(popt_gauss,np.diagonal(pcov_gauss))] + \
                ['{:.3f}'.format(pKS_g1)] + ['{:.3f}'.format(pKS_g2)] + \
                ['{:.3f} / {}'.format(chi2_gauss, NDOF_gauss)] + ['{:.3f}'.format(prob_gauss)]

    # ax.set_xlim(left=-0.3, right=0.3)
    ax.errorbar(x, y, yerr=sy, xerr=0, fmt='.', elinewidth=1)
    ax.plot(x, y_func, '--', label='Gaussian')
    if textpos == 'l':
        ax.text(0.02,0.98, nice_string_output(namesl, valuesl),
                    family='monospace',  transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', alpha=0.5)
    else:
        ax.text(0.6,0.98, nice_string_output(namesl, valuesl),
                    family='monospace',  transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', alpha=0.5)
    if legend:
        ax.legend(loc='center left')
    return ax
