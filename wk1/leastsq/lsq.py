import numpy as np
from scipy.optimize import minimize, curve_fit

# Code a chi square function?

def chisq(exp, data, unc):
    return ( np.power(exp - data,2) / np.power(unc,2) )


## first try scipy



# ## function to fit different order
#
# def fit_func(x, y, order):
#
#     pass
