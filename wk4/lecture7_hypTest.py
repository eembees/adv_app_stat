import numpy as np
from lib.nice_string_output import nice_string_output
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

def llh(data, pdf):
    return (np.sum(np.log(pdf(data)), axis=0))

def make_pdf(alfa, beta):
    # Return a function depending only on x.
    return lambda x: theoretical_pdf(x, alfa, beta)




def make_target(data):
    return lambda alfa, beta: -2 * llh(data, make_pdf(alfa, beta))

def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )


def bin_data(x, N_bins=100, range_x = None):
    if range_x == None:
        range_x = ( x.min(), x.max() )
    hist_y, hist_edges = np.histogram(x, bins=N_bins, range=range_x, density=True)
    hist_x = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_sy = np.sqrt(hist_y)
    hist_mask = hist_y > 0
    return hist_x, hist_y, hist_sy, hist_mask



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


        # Data histogram
        bin_centers, counts, sy, mask = bin_data(dat, N_bins=n_bins, range_x=range_x)



        x = bin_centers[mask]
        y = counts[mask]

        ax[i].errorbar(x, y, yerr=0, xerr=0, fmt='.', elinewidth=1)

        # Fit function

        y_func = theoretical_pdf(x, alfa, beta)

        ax[i].plot(x, y_func, '--', label='fit: a = {:.2f}, b = {:.2f}'.format(alfa, beta))


        ax[i].legend()



    plt.show()




