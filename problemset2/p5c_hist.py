import numpy as np
from scipy.integrate import quad
from pathlib import Path
from iminuit import Minuit
from lib.plot_gaussian import plot_gaussian
import lib.lib_math as lm
import scipy.stats
import matplotlib.pyplot as plt

# functions

def integrand(t_p, t, b, s):
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
        func_params = (t0, b, sigma)
        f = quad(integrand, 0, np.inf,
                 args=func_params)[0]
        result.append(f)
    return result


def make_pdf(b, sigma):
    def pdf(t):
        return expected_pdf(t, b, sigma)

    return pdf



# values from earlier
b_0 = 1.0
s_0 = 0.618
b_1 = 0.993
s_1 = 0.619

b_0_e = 0.0001
s_0_e = 0.00599
b_1_e = 0.00808
s_1_e = 0.0061

lamb = 1.358



nBins = 100

# data
f_data = list(Path.cwd().glob('*NucData.txt'))[0]

data = np.loadtxt(f_data)

range_x = ( data.min(), data.max() )

hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(data)

barwidth = (range_x[1] - range_x[0]) / nBins


# make pdf

pdf_h0 = make_pdf(b_0, s_0)
pdf_h1 = make_pdf(b_1, s_1)

# plotting arrays
xarr  = np.linspace(*range_x, num = 500)

yarr0 = pdf_h0(xarr)
yarr1 = pdf_h1(xarr)

# plot the best fit values
fig, ax = plt.subplots()

ax.bar(hist_x, hist_y, barwidth, label = 'Hist of values', alpha=0.5, color = 'xkcd:dark red')

ax.plot(xarr, yarr0,
        # ls = 'steps',
        label= 'Null hypothesis',
        color='red',
        lw=0.5,
        )
ax.plot(xarr, yarr1,
        # ls = 'steps',
        label= 'Test hypothesis',
        color='blue',
        lw=0.5,
        )

ax.set_title('Hypothesis testing - sanity test for all particle measurements')

ax.legend(loc='lower right')


# include distribution of mc results
ins = ax.inset_axes([4, 0.12, 2, 0.3], transform=ax.transData)
ins2 = ax.inset_axes([8, 0.12, 2, 0.3], transform=ax.transData)

ins.errorbar([1,2], [b_0, b_1], yerr=[b_0_e, b_1_e], fmt='None', label='B values', ecolor=['red','blue'], color=['red','blue'])
ins.set_title('$\sigma_t$s')

ins2.errorbar([3,4], [s_0, s_1], yerr=[s_0_e, s_1_e], fmt='None', label='$\sigma_t$ values',ecolor=['red','blue'], color=['red','blue'])
ins2.set_title('$b$s')

labels = [
    '',
'b_0',
'b_1',
]
labels2 = [
    '',
's_0',
's_1',
]

ins.set_xticklabels(labels)
ins2.set_xticklabels(labels2)



fig.tight_layout()

fig.savefig('p5c_sanity.pdf')

print(1 - scipy.stats.chi2.cdf(lamb, 1))

