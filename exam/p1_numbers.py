import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.lib_math as lm
import lib.monte_carlo as mc
import lib.nice_string_output as ns

from inspect import signature

range_a = (-10,10)
range_b = (-10,10)
range_c = (4000,8000)

range_1 = (20,27)
range_2 = (-1,1)
range_3 = (1,22)
range_4 = (0,2.5)
range_5 = (1,10)


def f1(x, a):
    return (1/(x+5)*np.sin(a*x))

def f2(x, a):
    return (np.sin(a*x)+1)

def f3(x, a):
    return (np.sin(a*np.square(x)))

def f4(x, a):
    return (np.square(np.sin(a*x)))

def f5(x):
    return x*np.tan(x)

def f6(x,a,b):
    return ( 1 + a*x + b*np.square(x))

def f6(x,a,b):
    return ( a + b*x)

def f7(x,a,b,c):
    return ( np.sin(a*x) + c*np.exp(b*x) + 1 )

def f8(x,a,b):
    return -1*np.exp(np.square(x-a)/(2*np.square(b)))


bins = 100
data = np.loadtxt('Exam_2019_Prob1.txt')

# # make hist and establish ranges
# fig, ax = plt.subplots(nrows=5, figsize=(4,10))
#
# for i in range(5):
#     range_dat = (data[:,i].min(), data[:,i].max())
#     print('Col {}: range: {:+00.2f} -> {:+0.2f}'.format(i, *range_dat))
#     barwidth = (range_dat[1] - range_dat[0]) / bins
#
#     hist_x, hist_y, hist_sy, hist_mask = lm.bin_data(data[:,i], N_bins=bins)
#
#     ax[i].bar(hist_x, hist_y, barwidth, label='Hist of values', alpha=0.5, color='xkcd:dark red')
#     ax[i].set_title('Col{}'.format(i))
#
# fig.tight_layout()
#
# fig.savefig('./figs/p1_inithist.pdf')
#


# here plot all the functions

funcs = [
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
]


figf, axf = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
axf = axf.ravel()


# dummy abc vals
a_ = 3
b_ = 3
c_ = 4500

for i, f in enumerate(funcs):
    # check num of params in func
    fsig = signature(f)
    nparams = len(fsig.parameters)


    xarr = np.linspace(*range_1)
    if nparams == 1:
        yarr = f(xarr)
    elif nparams == 2:
        yarr = f(xarr, a_)
    elif nparams == 3:
        yarr = f(xarr, a_, b_)
    elif nparams == 4:
        yarr = f(xarr, a_, b_, c_)

    axf[i].plot(xarr,yarr)

    # test_points = mc.monte_carlo()

figf.savefig('./figs/p1_functions.pdf')