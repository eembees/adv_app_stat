import numpy as np
import scipy.stats
import scipy.integrate
from lib.nice_string_output import nice_string_output
from lib.lib_math import bin_data
from iminuit import Minuit
import matplotlib.pyplot as plt

def get_normalization_constant(func, range_x):
    I = scipy.integrate.quad(func, *range_x)[0]
    return (1. / I)


def theoretical_function(x, alfa, beta):
    return (1 + alfa * x + beta * (x ** 2))


def theoretical_pdf(x, alfa, beta):
    range_mask = (x >= range_x[0]) & (x <= range_x[1])
    # C = 1 / (1.9 + 0.571583 * beta)  # Normalization constant from wolfram range +- 0.95
    C = 1 / (2/3 * (beta +3))  # Normalization constant from wolfram range +- 0.95
    return np.where(range_mask, C * theoretical_function(x, alfa, beta), 0)

def make_pdf(alfa, beta):
    # Return a function depending only on x.
    return lambda x: theoretical_pdf(x, alfa, beta)

def llh(data, pdf):
    return (np.sum(np.log(pdf(data)), axis=0))

def make_target(data):
    return lambda alfa, beta: -2 * llh(data, make_pdf(alfa, beta))



def theoretical_function_new(x, alfa, beta, gamma):
    return (1 + alfa * x + beta * (x ** 2) - gamma * (x**5))



def theoretical_pdf_new(x, alfa, beta, gamma):
    range_mask = (x >= range_x[0]) & (x <= range_x[1])
    # C = 1 / (1.9 + 0.571583 * beta)  # Normalization constant from wolfram range +- 0.95
    C = 1 / (2/3 * (beta +3))  # Normalization constant from wolfram range +- 0.95
    return np.where(range_mask, C * theoretical_function_new(x, alfa, beta, gamma), 0)


def make_pdf_new(alfa, beta, gamma):
    # Return a function depending only on x.
    return lambda x: theoretical_pdf_new(x, alfa, beta, gamma)


def make_target_new(data):
    return lambda alfa, beta, gamma: -2 * llh(data, make_pdf_new(alfa, beta, gamma))


def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )





if __name__ == '__main__':

    # Define params
    range_x = (-1, 1)

    n_bins = 100

    # Read data

    data1 = np.loadtxt('../wk4/Lecture8_LLH_Ratio_2_data.txt', delimiter=' ', usecols=[0])
    data2 = np.loadtxt('../wk4/Lecture8_LLH_Ratio_2a_data.txt', delimiter=' ', usecols=[0])

    data = [data1,data2]

    # # Fit the data to the pdf
    # minuit_LH1 = Minuit(make_target(data1), print_level=0, fix_alfa=False)
    # minuit_LH1.migrad()
    #
    # if (not minuit_LH1.get_fmin().is_valid):  # Check if the fit converged
    #     print("  WARNING: The (unbinned) likelihood fit DID NOT converge!!!")
    #
    # alfa1, beta1 = minuit_LH1.args
    #
    # # Fit the data to the pdf
    # minuit_LH2 = Minuit(make_target(data2), print_level=0, fix_alfa=False)
    # minuit_LH2.migrad()
    #
    # if (not minuit_LH2.get_fmin().is_valid):  # Check if the fit converged
    #     print("  WARNING: The (unbinned) likelihood fit DID NOT converge!!!")
    #
    # alfa2, beta2 = minuit_LH2.args


    # Make plots
    fig, axes = plt.subplots(nrows=2)
    ax = axes.ravel()

    for i, dat in enumerate(data):

        # Fit the data to the pdf
        minuit_LH1 = Minuit(make_target(dat), print_level=0, fix_alfa=False)
        minuit_LH1.migrad()

        if (not minuit_LH1.get_fmin().is_valid):  # Check if the fit converged
            print("WARNING: The (unbinned) likelihood fit DID NOT converge!!!")

        alfa, beta = minuit_LH1.args


        # Fit the data to new pdf
        minuit_LH2 = Minuit(make_target_new(dat), print_level=0, fix_alfa=False)
        minuit_LH2.migrad()

        if (not minuit_LH2.get_fmin().is_valid):  # Check if the fit converged
            print("WARNING: The (unbinned) likelihood fit DID NOT converge!!!")

        alfa_new, beta_new, gamma_new = minuit_LH2.args



        # Data histogram
        bin_centers, counts, sy, mask = bin_data(dat, N_bins=n_bins, range_x=range_x)



        x = bin_centers[mask]
        y = counts[mask]

        ax[i].errorbar(x, y, yerr=sy, xerr=0, fmt='.', elinewidth=1, c='xkcd:forest green')

        # Fit function

        x_plot = np.linspace(start = -1, stop = 1, num = 1000)

        y_func = theoretical_pdf(x_plot, alfa, beta)
        y_func_new = theoretical_pdf_new(x_plot, alfa_new, beta_new, gamma_new)

        ax[i].plot(x_plot, y_func, '--',c='xkcd:hot pink', label='fit: a = {:.3f}, b = {:.3f}'.format(alfa, beta))
        ax[i].plot(x_plot, y_func_new, '--',c='xkcd:red', label='fit: a = {:.3f}, b = {:.3f}, g = {:.3f}'.format(alfa_new, beta_new, gamma_new))


        ax[i].legend()

        # calculate llh values
        llh_old = llh(dat, make_pdf(alfa, beta))
        llh_new = llh(dat, make_pdf_new(alfa_new, beta_new, gamma_new))
        llh_rat = -2 * (llh_old - llh_new)
        pval    = scipy.stats.chi2.sf(llh_rat, 1)


        names = [
            'LLH H0',
            'LLH H1',
            'LLH ratio',
            'pval',
        ]

        values = ['{:.3f}'.format(val) for val in [llh_old, llh_new, llh_rat, pval]]

        print(nice_string_output(names, values))




    fig.tight_layout()
    plt.show()




