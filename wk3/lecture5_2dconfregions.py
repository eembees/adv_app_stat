# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T10:48:36+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: lecture3_exc2.py
# @Last modified by:   mag
# @Last modified time: 2019-02-13T11:22:41+01:00
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.monte_carlo import monte_carlo


def get_normalization_constant(func, range_x):
    I = scipy.integrate.quad(func, *range_x)[0]
    return (1. / I)


def theoretical_function(x, alfa, beta):
    return (1 + alfa * x + beta * (x ** 2))


def theoretical_pdf(x, alfa, beta):
    range_mask = (x >= range_x[0]) & (x <= range_x[1])
    C = 1 / (1.9 + 0.571583 * beta)  # Normalization constant from wolfram
    return np.where(range_mask, C * theoretical_function(x, alfa, beta), 0)


def make_pdf(alfa, beta):
    # Return a function depending only on x.
    return lambda x: theoretical_pdf(x, alfa, beta)


def llh(data, pdf):
    return (np.sum(np.log(pdf(data)), axis=0))


def make_target(data):
    return lambda alfa, beta: -2 * llh(data, make_pdf(alfa, beta))


def hist_get_cf(counts, bin_edges, sigmas):
    """

    :param counts: returned from np.histogram function
    :param bin_edges: also returned from np.histogram function
    :param sigmas: number of sigmas to calculcate confidence interval with
    :return: (cf_lft, cf_rgt) tuple of confidence interval
    """

    thres_lft = scipy.stats.norm.cdf(x=-sigmas)
    thres_rgt = scipy.stats.norm.cdf(x=sigmas)

    ECDF = np.nancumsum(counts * np.diff(bin_edges), axis=0)

    if ECDF[-1] != 1.0:  # if not normalized for some reason
        ECDF = ECDF / ECDF[-1]

    cf_lft = bin_edges[np.nonzero(np.diff(ECDF > thres_lft))]  # Works fine
    cf_rgt = bin_edges[np.nonzero(np.diff(ECDF < thres_rgt))[0] + 1]  # Doesnt work for very precise data

    return (cf_lft, cf_rgt)

def locllh(x,a,b, pdf):
    return (np.sum(np.log(pdf(x, a, b)),axis=0))


def raster_scan(data, range_a, range_b, pdf=theoretical_pdf, ax, dims = 50, plot_llh_contours = False):
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



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # stops minuit from bitching

    # define parameters
    alfa_true = 0.5
    beta_true = 0.5
    range_x = (-0.95, 0.95)

    boot_num = 500

    data_num = 2000

    n_bins = 100

    cf_int_sigmas = 1


    # define_func_true()
    def func_true(x):
        return theoretical_pdf(x, alfa_true, beta_true)


    # Make Some data
    data = monte_carlo(func_true, num=data_num, range_x=range_x)

    fig, ax = plt.subplots(figsize = (5,5))


    ax = raster_scan(data, range_a=(0,1.), range_b=(0.,2), pdf=func_true, axplot=ax, dims=100)


    plt.show()