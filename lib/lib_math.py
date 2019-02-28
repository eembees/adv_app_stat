import scipy.stats
import numpy as np

def critical_chi_sq(sigma, ndof):
    return scipy.stats.chi2.isf(1 - (0.5 - scipy.stats.norm.sf(sigma)) * 2, ndof)

def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )

def bin_data(x, N_bins=100, range_x = None):
    """
    Construct a normalized histogram of binned data with poisson uncertainties.

    :param x: ndarray
    :type x: np.array
    :param N_bins: number of bins
    :type N_bins: int
    :param range_x:
    :type range_x: tuple or None
    :return: Histogram locations, values, uncertainties, mask
    """
    if range_x == None:
        range_x = ( x.min(), x.max() )

    hist_y, hist_edges = np.histogram(x, bins=N_bins, range=range_x, density=False)
    hist_sy = np.sqrt(hist_y)
    hist_max_counts = hist_y.max()
    hist_y, hist_edges = np.histogram(x, bins=N_bins, range=range_x, density=True)
    hist_sy = hist_sy * (hist_y.max() / hist_max_counts)
    hist_x = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_mask = hist_y > 0
    return hist_x, hist_y, hist_sy, hist_mask

