import numpy as np
import matplotlib.pyplot as plt
import nestle as ne
import seaborn as sns

import lib.lib_math as lm



def llh_eggbox(t1, t2):
    """
    :param t1: angle in radians
    :param t2: angle in radians

    :type t2: float
    :type t1: float
    """
    return np.cos(t1/(3*np.pi)) * np.cos(t2/(3*np.pi))

def llh_eggbox_array(t):
    """
    :param t: angles in radians

    :type t: np.ndarray
    """
    return (np.cos(t[0]*15) * np.cos(t[1]*15))**3


xx, yy = np.meshgrid(np.linspace(0., 1., 100),
                     np.linspace(0., 1., 100))



# zip_xxyy = np.array([i for i in zip(xx.flatten(),yy.flatten())])
# Z = np.array([llh_eggbox(x, y) for (x, y) in zip_xxyy ]).reshape(xx.shape)



Z = llh_eggbox_array(np.array([xx,yy]))

# find things
res = ne.sample(llh_eggbox_array, lambda x:x, ndim=2, npoints = 500, method='multi',update_interval=20)


# with plt.xkcd():
fig, ax = plt.subplots(figsize=(10,6))

ax.contourf(xx, yy, Z, 12, cmap=plt.cm.Blues_r)

ax.scatter(x = res.samples[:,0].flatten(), y = res.samples[:,1].flatten(),c='xkcd:hot pink', marker = '+')


plt.show()

# g = sns.jointplot(x = res.samples[:,0], y = res.samples[:,1],
#         marginal_kws = dict(bins=50, rug=True),
#         )
#
# g = sns.jointplot(x = res.samples[:,0], y = res.samples[:,1], kind="hex",
#         marginal_kws = dict(bins=50, rug=True),
#         )
plt.show()