import numpy as np
from scipy.integrate import quad
from pathlib import Path
from iminuit import Minuit
from lib.plot_gaussian import plot_gaussian
import lib.lib_math as lm
import scipy.stats
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore")

#
# def make_integrand(t,b,s):
#     def integrand(t_p):
#         return np.exp(-(t - t_p) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s) * (np.exp(-t_p / b) / b)
#     return integrand

def integrand(t_p,t,b,s):
    return np.exp(-(t - t_p) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s) * (np.exp(-t_p / b) / b)

# def expected_pdf(t, b, sigma):
#     f = quad( make_integrand(t,b,sigma), 0, np.inf )[0]
#     return f

def expected_pdf(t, b, sigma):
    # Make t iterable if it is only a float/int
    if not hasattr(t, '__iter__'):
        t = np.array([t])

    result = []

    for t0 in t:
        # func_params = {
        #     't' : t0,
        #     'b' : b,
        #     'sigma':sigma
        # }
        func_params = ( t0,b, sigma )
        f = quad( integrand, 0, np.inf,
              args = func_params)[0]
        result.append(f)
    return result



def make_pdf(b,sigma):
    def pdf(t):
        return expected_pdf(t, b, sigma)
    return pdf

def make_target(data):
    return lambda b, sigma: -2 * lm.llh(data, make_pdf(b, sigma))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# set pars
init_sigma = 0.7
step_sigma = 0.1
init_b     = 1.0
step_b     = 0.5

# import data

f_data = list(Path.cwd().glob('*NucData.txt'))[0]

data = np.loadtxt(f_data)

data_chunks_gen = chunks(data, 200)

lh_ratios = []

counter = 1
for data_chunk in data_chunks_gen:
    print('Experiment number {}'.format(counter))
    counter += 1

    # null hypothesis
    minuit_LH0 = Minuit(make_target(data_chunk), print_level=0, b = init_b,sigma=init_sigma, fix_b=True,pedantic=False)
    minuit_LH0.migrad()


    # not null hypothesis
    minuit_LH1 = Minuit(make_target(data_chunk), print_level=0, b=init_b, sigma=init_sigma, pedantic=False)
    minuit_LH1.migrad()



    if (not minuit_LH0.get_fmin().is_valid):  # Check if the fit converged
        print("WARNING: The (unbinned) likelihood fit DID NOT converge! (H0)")
        llh_H0 = None
    else:
        llh_H0 = minuit_LH0.fval
        # print(minuit_LH0.args[1])
    if (not minuit_LH1.get_fmin().is_valid):  # Check if the fit converged
        print("WARNING: The (unbinned) likelihood fit DID NOT converge! (H1)")
        llh_H1 = None
    else:
        llh_H1 = minuit_LH1.fval
        # print(minuit_LH1.args[1])

    if (llh_H0 is not None) and (llh_H1 is not None ):
        llh_rat = 2 * (llh_H0 - llh_H1)

        lh_ratios.append(llh_rat)



lh_ratios = np.array(lh_ratios)
print(lh_ratios)
# now we can work on this separately. that will make my poor computer very happy
np.savetxt('p5_llhratios.txt', lh_ratios)

# hist_tuple = lm.bin_data(lh_ratios, bins=20)

fig, ax = plt.subplots()

# ax.errorbar(hist_tuple[0], hist_tuple[1])

ax = plot_gaussian(lh_ratios, ax = ax, nBins=25)

plt.savefig('p5_llh.pdf')
