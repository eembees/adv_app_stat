import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
# Coin flipping bias

def coinflipping(ax):
    llh   = lambda x: scipy.stats.binom.pmf(heads, n = n, p = x)

    prior = lambda x : scipy.stats.beta(a, b).pdf(x)

    posterior = lambda x : (llh(x) * prior(x))



    # Plot priors. llh, posterios, n=100 flips, k=66
    xarr = np.linspace(0,1,num=1000)

    yarr_prior = prior(xarr)
    yarr_prior = yarr_prior / yarr_prior.max()

    yarr_llh = llh(xarr)
    yarr_llh = yarr_llh  / yarr_llh.max()

    yarr_post = posterior(xarr)
    yarr_post = yarr_post / yarr_post.max()



    ax.plot(xarr, yarr_prior, label = 'Prior')
    ax.plot(xarr, yarr_llh, label = 'LLH')
    ax.plot(xarr, yarr_post, label = 'post')


    ax.legend()

    pass

if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=2,figsize = (6,10))

    n = 100
    heads = 66

    a = 5
    b = 17

    coinflipping(ax[0])

    n = 1000
    heads = 660

    coinflipping(ax[1])


    fig.tight_layout()
    plt.show()


