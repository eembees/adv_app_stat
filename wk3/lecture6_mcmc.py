import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import lib.plot_gaussian

import pomegranate as pg


# Coin flipping bias

def MH(ax, ax2, ax3, n_iter = 2000, n = 100, heads = 66, a = 5, b = 17):
    def llh(x):
        return scipy.stats.binom.pmf(heads, n = n, p = x)

    def prior(x) :
        return scipy.stats.beta(a, b).pdf(x)

    def posterior(x ):
        return (llh(x) * prior(x))

    def proposal(x):
        # pdf_val = scipy.stats.norm.rvs(0,0.1)
        pdf_val = np.random.normal(loc = 0.0, scale = 0.1)
        return x + pdf_val

    post_xs      = np.arange(n_iter)
    post_samples = np.zeros(n_iter)

    post_samples[0] = 1 # set a nice initial value


    for i in range(n_iter-1):
        theta_old = post_samples[i]

        theta_new = proposal(theta_old)

        # prob_accept = (llh(theta_new) / llh(theta_old))
        prob_accept = (posterior(theta_new) / posterior(theta_old))


        # generate a random number u
        u = np.random.uniform()

        if prob_accept >= 1.0:
            # Accept automatically
            post_samples[i+1] = theta_new
        elif prob_accept > u :
            # Accept if random uniform num is larger than acceptance probability
            post_samples[i+1] = theta_new
        else :
            post_samples[i+1] = theta_old


    # ax.scatter(post_xs, post_samples,marker='2', color = 'xkcd:hot pink', label = 'MH MCMC samples')
    ax.plot(post_xs, post_samples,'2',linestyle = '--', linewidth=1, color = 'xkcd:hot pink', label = 'MH MCMC samples')

    ax2 = lib.plot_gaussian.plot_gaussian(post_samples, ax=ax2)
    ax3 = lib.plot_gaussian.plot_gaussian(post_samples[ n_iter//10 :], ax=ax3) # burned out gaussian

    ax2.set_xlim(0,1)

    ax.set_ylim(-0.5,1.5)

    ax.legend()

    pass


def MH_pg(ax,ax2, n_iter = 2000, n = 100, heads = 66):
    pass


if __name__ == '__main__':

    fig, axes = plt.subplots(nrows=2,ncols=2,figsize = (10,10),sharex='col')
    ax = axes.ravel()
    n = 100
    heads = 66

    a = 5
    b = 17

    MH(ax[0], ax[1], ax[3], n_iter=5000)




    fig.tight_layout()
    plt.show()
    exit()


