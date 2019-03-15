import numpy as np
from scipy.integrate import quad
from pathlib import Path
from iminuit import Minuit
from lib.plot_gaussian import plot_gaussian
import lib.lib_math as lm
import scipy.stats
import matplotlib.pyplot as plt


# import warnings
# warnings.filterwarnings("ignore")

#
# def make_integrand(t,b,s):
#     def integrand(t_p):
#         return np.exp(-(t - t_p) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s) * (np.exp(-t_p / b) / b)
#     return integrand

def integrand(t_p, t, b, s):
    return np.exp(-(t - t_p) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s) * (np.exp(-t_p / b) / b)


# def expected_pdf(t, b, sigma):
#     f = quad( make_integrand(t,b,sigma), 0, np.inf )[0]
#     return f

def expected_pdf(t, b, sigma):
    # Make t iterable if it is only a float/int
    if not hasattr(t, '__iter__'):
        t = np.array([t])

    result = []

    for t0 in t:
        # func_params = {
        #     't' : t0,
        #     'b' : b,
        #     'sigma':sigma
        # }
        func_params = (t0, b, sigma)
        f = quad(integrand, 0, np.inf,
                 args=func_params)[0]
        result.append(f)
    return result


def make_pdf(b, sigma):
    def pdf(t):
        return expected_pdf(t, b, sigma)

    return pdf


def make_target(data):
    return lambda b, sigma: -2 * lm.llh(data, make_pdf(b, sigma))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# set pars
init_sigma = 0.7
step_sigma = 0.1
init_b = 1.0
step_b = 0.5

# import data

f_data = list(Path.cwd().glob('*NucData.txt'))[0]

data = np.loadtxt(f_data)

# null hypothesis
minuit_LH0 = Minuit(make_target(data), print_level=2, b=init_b, sigma=init_sigma, fix_b=True, pedantic=True)
minuit_LH0.migrad()

# not null hypothesis
minuit_LH1 = Minuit(make_target(data), print_level=2, b=init_b, sigma=init_sigma, pedantic=True)
minuit_LH1.migrad()

llh_H0 = minuit_LH0.fval
llh_H1 = minuit_LH1.fval

lambda_x = -2 * (llh_H0 - llh_H1)

print(llh_H0)
print(llh_H1)
print(lambda_x)

'''
**************************************************************************************
**************************************************
*                     MIGRAD                     *
**************************************************

VariableMetric: start iterating until Edm is < 0.0002
VariableMetric: Initial state   - FCN =   59470.06960764 Edm =      199.781 NCalls =     13
VariableMetric: Iteration #   0 - FCN =   59470.06960764 Edm =      199.781 NCalls =     13
VariableMetric: Iteration #   1 - FCN =   59308.66483824 Edm =      3.54454 NCalls =     19
VariableMetric: Iteration #   2 - FCN =   59305.48571934 Edm =   0.00537371 NCalls =     24
VariableMetric: Iteration #   3 - FCN =   59305.48118145 Edm =  5.92986e-05 NCalls =     30
VariableMetric: After Hessian   - FCN =   59305.48118145 Edm =  5.65202e-05 NCalls =     40
VariableMetric: Iteration #   4 - FCN =   59305.48118145 Edm =  5.65202e-05 NCalls =     40
**************************************************************************************
--------------------------------------------------------------------------------------
fval = 59305.481181446165 | total call = 40 | ncalls = 40
edm = 5.652021702492786e-05 (Goal: 1e-05) | up = 1.0
--------------------------------------------------------------------------------------
|          Valid |    Valid Param | Accurate Covar |         Posdef |    Made Posdef |
--------------------------------------------------------------------------------------
|           True |           True |           True |           True |          False |
--------------------------------------------------------------------------------------
|     Hesse Fail |        Has Cov |      Above EDM |                |  Reach calllim |
--------------------------------------------------------------------------------------
|          False |           True |          False |                |          False |
--------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
| No | Name  |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
----------------------------------------------------------------------------------------
|  0 |     b | 0.993    | 0.00809  |          |          |          |          |       |
|  1 | sigma | 0.619    | 0.00609  |          |          |          |          |       |
----------------------------------------------------------------------------------------
**************************************************************************************
59306.16020910043
59305.481181446165
-1.3580553085339488

Process finished with exit code 0

'''
