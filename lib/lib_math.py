import scipy.stats
import numpy as np

def critical_chi_sq(sigma, ndof):
    return scipy.stats.chi2.isf(1 - (0.5 - scipy.stats.norm.sf(sigma)) * 2, ndof)

def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )

def llh(data, pdf):
    return (np.sum(np.log(pdf(data)), axis=0))


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



def make_KDE(x:np.ndarray, h: float = 1.5, type: str = 'gauss'):
    """
    :param x : data to construct KDE from
    :type x : np.ndarray
    :param h : height
    :type h : float

    :rtype: object
    """
    # THE KERNEL IS ALWAYS NORMALIZED

    global K

    if type == 'flat':
        def K(x_0, x_1, h=h):
            w = 2 * h
            k = 1 / w
            diff = np.abs(x_0 - x_1)
            return k if diff <= w / 2 else 0
    elif type == 'gauss':
        def K(x_0, x_1, h=h):
            k = (1/(np.sqrt(2*np.pi)*h))*np.exp(-(np.abs(x_1-x_0)**2) / (2 * h**2) )
            return k
    elif type == 'epan':
        def K(x_0, x_1, h=h):
            u = np.abs(x_0 - x_1)/h
            k = 3/4 * ((1 - (u**2)))#/((1./2) * (3*h -1.) ))
            return k if k > 0 else 0

    # n_x = np.sum(x.shape)
    n_x = len(x)

    def pdf_KDE(xi):
        return (1 /(n_x)) * np.sum(
            [K(x_0=xj, x_1=xi)/h for xj in x]
        )

    return pdf_KDE
