# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T09:11:38+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: lecture3.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T09:56:29+01:00
import numpy as np


def gaussian_likelihood(mu, sigma, data):
    """Short summary.

    Parameters
    ----------
    mu : type
        Description of parameter `mu`.
    sigma : type
        Description of parameter `sigma`.
    data : type
        Description of parameter `data`.

    Returns
    -------
    type
        Description of returned object.

    """
    likelihoods = (1 / np.sqrt(2 * np.pi * sigma)) * \
        np.exp(-(data - mu)**2 / (2 * sigma))

    llh = np.prod(likelihoods)

    # print("Likelihood for the data")
    # print("Gaussian, sigma = {}, mu = {}".format(sigma, mu))
    # print('{:.5f}'.format(llh))
    # pass
    return llh

if __name__ == '__main__':
    x = np.array([
        1.01,
        1.3,
        1.35,
        1.44,
    ]
    )

    gaussian_likelihood(mu=1.25, sigma=0.11, data=x)
    gaussian_likelihood(mu=1.30, sigma=0.50, data=x)
