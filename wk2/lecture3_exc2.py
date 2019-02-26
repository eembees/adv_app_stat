# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T10:48:36+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: lecture3_exc2.py
# @Last modified by:   mag
# @Last modified time: 2019-02-13T11:22:41+01:00
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
from iminuit import Minuit
from probfit import UnbinnedLH

from monte_carlo import monte_carlo

# define parameters
alpha_true = 0.5
beta_true = 0.5
range_x = (-1.0, 1.0)

def get_normalization_constant(func, range_x):
    I = scipy.integrate.quad(func, *range_x)[0]
    return (1. / I)


def theoretical_function(x, alpha, beta):
    return (1 + alpha * x + beta * (x ** 2) / ())

def theoretical_pdf(x, alpha, beta):
    C = get_normalization_constant(lambda i : theoretical_function(i, alpha, beta), range_x)
    return C * theoretical_function(x, alpha, beta)


def define_func_true():
    global func_true
    def func_true(x):
        return theoretical_function(x, alpha_true, beta_true)
    C = get_normalization_constant(
            func_true,
            range_x)
    def func_true(x):
        return C * theoretical_function(x, alpha_true, beta_true)
    pass


define_func_true()

data = monte_carlo(func_true, num=2000, range_x=(-1, 1))





