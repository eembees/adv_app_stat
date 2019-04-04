import scipy.stats
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator

def critical_chi_sq(sigma, ndof):
    return scipy.stats.chi2.isf(1 - (0.5 - scipy.stats.norm.sf(sigma)) * 2, ndof)

def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )

def llh(data, pdf):
    return (np.nansum(np.log(pdf(data)), axis=0))


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
        return (1 /(h * n_x)) * np.sum(
            [K(x_0=xj, x_1=xi) for xj in x]
        )

    return pdf_KDE




def locllh(x,a,b, pdf):
    return (np.sum(np.log(pdf(x, a, b)),axis=0))


def raster_scan(data, range_a, range_b,  pdf, ax, dims = 50, plot_llh_contours = False):
    """
    Makes 2d raster scan for pdf in the specified ranges
    :param data:
    :param range_a:
    :param range_b:
    :param pdf: callable, must be of the forma f(x,a,b) for llh estimation
    :param ax:
    :param dims:
    :param plot_llh_contours:
    :return:
    """
    vals_a   = np.linspace(*range_a, num=dims)
    vals_b   = np.linspace(*range_b, num=dims)

    a, b = np.meshgrid(vals_a, vals_b, indexing='ij')

    x_3d = data[:, np.newaxis, np.newaxis]
    a_3d = a[np.newaxis, :, :]
    b_3d = b[np.newaxis, :, :]

    rst_llh = locllh(x_3d, a_3d, b_3d, pdf=pdf)

    ax.pcolormesh(a, b, rst_llh)

    rst_max_idx = np.nanargmax(rst_llh)
    a_idx, b_idx = np.unravel_index(rst_max_idx, rst_llh.shape)
    rst_max_b = vals_a[a_idx]
    rst_max_a = vals_b[b_idx]
    rst_max_llh = rst_llh[a_idx, b_idx]

    if plot_llh_contours == True:
        llh_1s = rst_max_llh - 0.5
        llh_2s = rst_max_llh - 2.
        llh_3s = rst_max_llh - 4.5

        llhs_s = [
            llh_1s,
            llh_2s,
            llh_3s,
        ]

        cf_colors = [
            'xkcd:dark purple',
            'xkcd:purple',
            'xkcd:light purple',
        ]


        ax.contour(a, b, rst_llh, levels=sorted(llhs_s),colors = cf_colors, label='CF')

        legend_elements = [Line2D([0], [0], color=cf_colors[0], label='LLH 3s'),
                           Line2D([0], [0], color=cf_colors[1], label='LLH 2s'),
                           Line2D([0], [0], color=cf_colors[2], label='LLH 1s'),]


        ax.legend(handles=legend_elements)


    return ax

def spline(xdata, ydata, kind='linear', num=10000, int_range=None):
    range_x = (xdata.min(),xdata.max())
    xnew = np.linspace(*range_x, num=num)
    if kind == 'pchip':
        f = PchipInterpolator(xdata,ydata)
    else:
        f = interp1d(xdata, ydata, kind=kind)
    if int_range is None:
        int = quad(f, *range_x)[0]
    else:
        int = quad(f, *int_range)[0]
    y = f(xnew)
    return(f,y,int)