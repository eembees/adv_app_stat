import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom, norm

K = 100  # number of marked fish
n = 60  # number of drawn fish
k = 10  # number of drawn marked fish

pN = 1
pk = 1

colors = [
    'xkcd:pastel blue',
    'xkcd:pastel yellow',
    'xkcd:pastel purple',
    'xkcd:pastel pink',
    'xkcd:pastel red',
    'xkcd:pastel orange',
]

N = np.arange(start=100, stop=1500, step=1, dtype=np.int_)

pkN = hypergeom.pmf(k - 1, N, K, n)

pNk = pkN * pN / pk  # Bayes Theorem


def make_pnk(N, K, n, k, pk=1, pN=None, normalize=False):
    '''
    returns probability of n given k
    :param N: ndarray of n values
    :param K: number of samples drawn
    :param n: number of known marked samples
    :param k: number of marked samples drawn
    :return pNk: hypergeometric probability
    '''

    if pN == 'inv':
        pN = 1 / N
    elif pN == 'flat':
        pN = 1
    elif callable(pN):
        pN = pN(N)
    else:
        pN = 1

    pkN = hypergeom.pmf(k - 1, N, K, n)

    pNk = pkN * pN / pk  # Bayes Theorem

    if normalize == True:
        pNk = pNk / np.sum(pNk)

    return pNk


def prior_gaussian(x, mean=500, sigma=35):
    return norm.pdf(x, loc=mean, scale=sigma)


def new_gaussian(x, mean=500, sigma=100):
    return norm.pdf(x, loc=mean, scale=sigma)


def extreme_gaussian(x, mean=500, sigma=61):
    return norm.pdf(x, loc=mean, scale=sigma)


# k=10 flat


pNk_k10_flat = make_pnk(N, K, n, k=10, normalize=True)
pNk_k15_flat = make_pnk(N, K, n, k=15, normalize=True)

pNk_k10_inv = make_pnk(N, K, n, k=10, pN='inv', normalize=True)
pNk_k15_inv = make_pnk(N, K, n, k=15, pN='inv', normalize=True)

pNk_gaussian_k4 = make_pnk(N, K=50, n=30, k=4, pN=prior_gaussian, normalize=True)
pNk_gaussian_k8 = make_pnk(N, K=50, n=30, k=8, pN=prior_gaussian, normalize=True)

pNk_doublegaussian_k4 = make_pnk(N, K=50, n=30, k=4, pN=new_gaussian, normalize=True)
pNk_doublegaussian_k8 = make_pnk(N, K=50, n=30, k=8, pN=new_gaussian, normalize=True)

pNk_extreme_gs = make_pnk(N, K=50, n=30, k=12, pN=extreme_gaussian, normalize=True)
pNk_extreme_LH = make_pnk(N, K=50, n=30, k=8, pN=1, normalize=True)

# normalize the whole thing

# pNk_k10_flat = pNk_k10_flat / np.nanmax(pNk_k10_flat)
# pNk_k15_flat = pNk_k15_flat / np.nanmax(pNk_k15_flat)
#
# pNk_k10_inv = pNk_k10_inv  / np.nanmax(pNk_k10_inv)
# pNk_k15_inv = pNk_k15_inv / np.nanmax(pNk_k15_inv)


# normalization ish
# pNk_k10_inv = pNk_k10_inv  * ( pNk_k10_flat / pNk_k10_inv.max() )
# pNk_k15_inv = pNk_k15_inv * ( pNk_k10_flat / pNk_k15_inv.max() )

# print(pNk_k10_flat)
# print(pNk_k10_inv)
# print(pNk_k15_flat)
# print(pNk_k15_inv)


plt.close('all')
fig, axes = plt.subplots(3, figsize=(6, 10))
ax = axes[0]
ax2 = axes[1]
ax3 = axes[2]

ax.set_title('Likelihood')
ax.plot(N, pNk_k10_flat, c=colors[3], ls='dotted',
        label='k=10 flat, max={}'.format(N[np.nanargmax(pNk_k10_flat)]))
ax.plot(N, pNk_k15_flat, c=colors[4], ls='dotted',
        label='k=15 flat, max={}'.format(N[np.nanargmax(pNk_k15_flat)]))
ax.plot(N, pNk_k10_inv, c=colors[3], ls='dashed',
        label='k=10 inv, max={}'.format(N[np.nanargmax(pNk_k10_inv)]))
ax.plot(N, pNk_k15_inv, c=colors[4], ls='dashed',
        label='k=15 inv, max={}'.format(N[np.nanargmax(pNk_k15_inv)]))


ax2.set_title('Gaussians')
ax2.plot(N, pNk_gaussian_k4, c=colors[0], ls='dashed',
         label='k=4, max={}'.format(N[np.nanargmax(pNk_gaussian_k4)]))
ax2.plot(N, pNk_gaussian_k8, c=colors[0], ls='dotted',
         label='k=88, max={}'.format(N[np.nanargmax(pNk_gaussian_k8)]))

ax2.plot(N, pNk_doublegaussian_k4, c=colors[5], ls='dashed',
         label='db k=4, max={}'.format(N[np.nanargmax(pNk_doublegaussian_k4)]))
ax2.plot(N, pNk_doublegaussian_k8, c=colors[5], ls='dotted',
         label='db k=8, max={}'.format(N[np.nanargmax(pNk_doublegaussian_k8)]))

ax2.set_title('Extreme')

ax3.plot(N, pNk_extreme_LH, c=colors[5], ls='dashed',
         label='ext|LH k=4, max={}'.format(N[np.nanargmax(pNk_extreme_LH)]))
ax3.plot(N, pNk_extreme_gs, c=colors[5], ls='dotted',
         label='ext|GS k=8, max={}'.format(N[np.nanargmax(pNk_extreme_gs)]))


ax.legend()
ax2.legend()
ax3.legend()

fig.tight_layout()

plt.show()
