import numpy as np
import matplotlib.pyplot as plt

from lsq import chisq

# my own libs
def nice_string_output(names, values, extra_spacing = 0,):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))
    return string[:-2]



def poly_fit(x, y, deg):
    lm = np.polyfit(x, y, deg)
    return np.poly1d(lm)

def fit_poly_fit(x_data, y_data, yerr, order, ax=None):
    f = poly_fit(x_data, y_data, order)
    ## calc chi sq
    expected = np.array([ f(x) for x in x_data ])
    chi2s = np.array([chisq(exp, y, yerr) for exp, y in zip(expected, y_data)])
    chi2  = np.sum(chi2s)

    if ax is not None:
        x_plotting = np.linspace(
        x_data[0],
        x_data[-1],
        num = 100
        )
        expected_plot = [ f(x) for x in x_plotting ]

        ## plotting
        ax.plot(
        x_plotting,
        expected_plot,
        marker = None,
        linestyle='dashed',
        alpha=0.7,
        label = '{}. Chi: {:.1f} '.format(order, chi2)
        )

    pass


def gen_fit_gaussian(n, ax, poly_orders=range(3)):

    ## Generate some data
    samples = np.random.normal(size=n)
    hist, bin_edges = np.histogram( samples, bins=100 )

    # get bin_width
    bin_width = bin_edges[1] - bin_edges[0]

    bin_centers = bin_edges[1:] - bin_width

    # Now, what if some hist entries are zeros?
    bin_centers = bin_centers[hist>0]
    hist = hist[hist>0]


    yerr = np.sqrt(hist)

    ax.errorbar(
    bin_centers,
    hist,
    yerr=yerr,
    fmt='.',
    ecolor='xkcd:hot pink',
    linewidth=0.1
    )

    for order in poly_orders:
        fit_poly_fit(bin_centers, hist, yerr, order, ax=ax)




    ax.legend()
    ax.set_title("Randomly generated numbers")
    pass

def exc1():
    x_data = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    ]

    y_data = [
    0.0,
    0.8,
    0.9,
    0.1,
    -0.8,
    -1.0,
    ]

    yerr = 0.5

    orders = 5

    fig, ax = plt.subplots()
    ax.errorbar(
    x_data,
    y_data,
    yerr=yerr,
    fmt='o',
    ecolor='xkcd:hot pink'
    )
    for order in range(orders):
        fit_poly_fit(x_data, y_data, yerr, order, ax=ax)


    ax.legend()
    fig.tight_layout()
    plt.show()
    pass


def exc2():
    ns=[10, 100, 1000, 10000, 100000]
    fig, ax = plt.subplots(2,2)
    axes = ax.ravel()

    for ax, n in zip(axes, ns):
        gen_fit_gaussian(n, ax, poly_orders=[6,12,36])

    fig.tight_layout()
    plt.show()

    pass


if __name__ == '__main__':
    # exc1()
    exc2()
