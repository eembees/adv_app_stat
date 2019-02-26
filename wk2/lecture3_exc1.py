# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T09:11:38+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: lecture3.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T10:16:12+01:00
import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import cm
import seaborn as sns
import pandas as pd


def gaussian_log_likelihood(mu, sigma, data):
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
    log_likelihoods = np.log((1 / np.sqrt(2 * np.pi * sigma**2)) * \
        np.exp(-(data - mu)**2 / (2 * sigma**2)))

    llh = np.sum(log_likelihoods)

    # print("Likelihood for the data")
    # print("Gaussian, sigma = {}, mu = {}".format(sigma, mu))
    # print('{:.5f}'.format(llh))
    # pass
    return llh

if __name__ == '__main__':

    mu_true    = 0.2
    sigma_true = 0.1

    data = np.random.normal(loc=0.2, scale=0.1, size = 100)

    mu_guess_arr = np.linspace(0.01,0.5,num=10)
    sigma_guess_arr = np.linspace(0.01,0.5,num=10)

    xv, yv = np.meshgrid(mu_guess_arr, sigma_guess_arr)


    xv_temp = xv.copy()

    xv = xv.flatten()
    yv = yv.flatten()

    zv = []

    for mu_guess, sigma_guess in zip(xv, yv):
        llh = gaussian_log_likelihood(mu_guess, sigma_guess, data)
        zv.append(llh)

    # make df
    df = pd.DataFrame.from_dict(
    np.array([xv,yv,zv]).T
    )
    df.columns = ['X_value','Y_value','Z_value']

    pivotted= df.pivot('Y_value','X_value','Z_value')

    sns.heatmap(pivotted,cmap='RdBu')



    # zv = np.reshape(np.array(zv), xv_temp.shape )
    #
    #
    # fig, ax = plt.subplots()
    #
    # ax.pcolormesh(zv)

    # ax.scatter(xv,yv,s=20,c=zv, marker = 'o', cmap = cm.plasma );



    plt.show()
