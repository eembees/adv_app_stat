import numpy as np
import matplotlib.pyplot as plt

from lib.nice_string_output import nice_string_output

def basic_KDE(x:np.ndarray, h: float = 1.5):
    """
    :param x : data to construct KDE from
    :type x : np.ndarray
    :param h : height
    :type h : float

    :rtype: object
    """
    # THE KERNEL IS ALWAYS NORMALIZED

    def K(x_0,x_1, h=h):
        w = 2*h
        k = 1/w
        diff = np.abs(x_0-x_1)
        return k if diff <= w/2 else 0

    n_x = np.sum(x.shape)

    def pdf_KDE(xi):
        return 1 / n_x * np.sum([K(x_0=xj, x_1=xi) for xj in x])

    return pdf_KDE





data = np.array([1,2,5,6,12,15,16,16,22,22,22,23])



hlist = [0.5,1,1.5]


fig, axes = plt.subplots(nrows=3, figsize=(6,10))
ax=axes.ravel()


for i, ax in enumerate(axes):
    kde = basic_KDE(x=data, h=hlist[i])

    x_plot = np.linspace(start=data.min() - 1, stop=data.max() + 1, num=200)
    y_plot = [kde(x_y) for x_y in x_plot]

    testdata = [6, 10.1, 20.499, 20.501]

    names = ['{:.3f}'.format(dat) for dat in testdata]
    values = ['{:.3f}'.format(kde(dat)) for dat in testdata]

    names.reverse()
    values.reverse()

    names.append('Xval')
    values.append('Yval')

    names.reverse()
    values.reverse()

    ax.scatter(data, np.zeros_like(data))

    ax.plot(x_plot, y_plot)

    ax.text(0.02,0.98, nice_string_output(names, values), family='monospace',
                transform=ax.transAxes,
                fontsize=10, verticalalignment='top', alpha=0.5)

fig.tight_layout()

plt.show()