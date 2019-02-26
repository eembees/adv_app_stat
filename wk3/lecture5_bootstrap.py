# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T10:48:36+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: lecture3_exc2.py
# @Last modified by:   mag
# @Last modified time: 2019-02-13T11:22:41+01:00
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import scipy.stats
from iminuit import Minuit


from lib.monte_carlo import monte_carlo



def get_normalization_constant(func, range_x):
    I = scipy.integrate.quad(func, *range_x)[0]
    return (1. / I)


def theoretical_function(x, alfa, beta):
    return (1 + alfa * x + beta * (x ** 2))

def theoretical_pdf(x, alfa, beta):
    range_mask = (x >= range_x[0]) & (x <= range_x[1])
    C = 1 / (1.9 + 0.571583 * beta) # Normalization constant from wolfram
    return np.where(range_mask, C * theoretical_function(x, alfa, beta), 0)


def make_pdf(alfa, beta):
    # Return a function depending only on x.
    return lambda x: theoretical_pdf(x, alfa, beta)

def llh(data, pdf):
    return(np.sum(np.log(pdf(data)), axis=0))

def make_target(data):
    return lambda alfa, beta: -2*llh(data, make_pdf(alfa, beta))

def hist_get_cf(counts, bin_edges, sigmas):
    """

    :param counts: returned from np.histogram function
    :param bin_edges: also returned from np.histogram function
    :param sigmas: number of sigmas to calculcate confidence interval with
    :return: (cf_lft, cf_rgt) tuple of confidence interval
    """

    thres_lft = scipy.stats.norm.cdf(x=-sigmas)
    thres_rgt = scipy.stats.norm.cdf(x= sigmas)

    ECDF = np.nancumsum(counts*np.diff(bin_edges), axis=0)

    if ECDF[-1] != 1.0: # if not normalized for some reason
        ECDF = ECDF / ECDF[-1]

    cf_lft = bin_edges[np.nonzero( np.diff( ECDF > thres_lft ) ) ] # Works fine
    cf_rgt = bin_edges[np.nonzero( np.diff( ECDF < thres_rgt ) )[0] + 1 ] # Doesnt work for very precise data

    return(cf_lft, cf_rgt)



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning) # stops minuit from bitching
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


    alfas = np.zeros(shape=boot_num)
    betas  = np.zeros(shape=boot_num)

    for i in range(boot_num):
        data = monte_carlo(func_true, num=data_num, range_x=range_x)

        minuit_LH = Minuit(make_target(data),print_level=0, alfa=0.5, fix_alfa=True)
        minuit_LH.migrad()

        if (not minuit_LH.get_fmin().is_valid):  # Check if the fit converged
            print("  WARNING: The (unbinned) likelihood fit DID NOT converge!!!")

        alfa_temp, beta_temp = minuit_LH.args

        alfas[i] = alfa_temp
        betas[i] = beta_temp

    # find confidence intervals

    ## Define integration constant

    ## Make ECDF

    counts_alfas, bin_edges_alfas = np.histogram(alfas, bins = n_bins, density = True, range = (0,1) )
    counts_betas, bin_edges_betas = np.histogram(betas, bins = n_bins, density = True, range = (0,1) )


    # Find where ECDF crosses threshold
    cf_alfa = hist_get_cf(counts_alfas, bin_edges_alfas, cf_int_sigmas)
    cf_beta = hist_get_cf(counts_betas, bin_edges_betas, cf_int_sigmas)

    print(cf_alfa)
    print(cf_beta)
