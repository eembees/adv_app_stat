import numpy as np
import matplotlib.pyplot as plt

from lib.nice_string_output import nice_string_output

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
            diff = np.abs(x_0 - x_1)
            k = 3/4 * (h - (diff**2))
            return k if k > 0 else 0

    n_x = np.sum(x.shape)

    def pdf_KDE(xi):
        return 1 / n_x * np.sum(
            [K(x_0=xj, x_1=xi) for xj in x]
        )

    return pdf_KDE


data = np.array([1,2,5,6,12,15,16,16,22,22,22,23])



hlist = [0.5,5]
typelist = [
    'flat',
    'gauss',
    'gauss',
    'epan',
]

fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(15,10))
ax=axes.ravel()
# x_plot = np.linspace(start=data.min() - 1, stop=data.max() + 1, num=200)
x_plot = np.linspace(start=-10, stop=35, num=200)

for i, axi in enumerate(ax):
    kde = make_KDE(x=data, h=hlist[1], type=typelist[i])

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

    # axi.scatter(data, np.zeros_like(data), c='xkcd:hot pink')

    axi.plot(x_plot, y_plot)
    axi.plot(data, np.full_like(data, -0.5), '|',c='xkcd:purple', markeredgewidth=1)

    axi.text(0.02, 0.98, nice_string_output(names, values), family='monospace',
             transform=axi.transAxes,
             # fontsize=10,
             verticalalignment='top', alpha=0.5)

    # Now plot the individual kernels
    for dat in data:
        y_k = [K(dat, x_y) for x_y in x_plot]
        axi.plot(x_plot, y_k, c='xkcd:vermillion', lw=1, ls='--')



fig.tight_layout()

plt.show()