import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.lib_math as lm
import lib.nice_string_output as ns
import corner

from iminuit import Minuit
import nestle as ne

sq2pi = np.sqrt(2 * np.pi)
t1_range = (0, 7*np.pi)
t2_range = (0, 7*np.pi)
t3_range = (0, 3)

tw = np.array([
    t1_range[1],
    t2_range[1],
    t3_range[1],
])

ranges = [
    t1_range,
    t2_range,
    t3_range
]


def L(t1, t2, t3, sigmasq=0.04, mu=0.68):
    sigma = np.sqrt(sigmasq)
    return np.cos(t1) * np.cos(t2) + (1 / (sigma * sq2pi)) * np.exp(
        - (np.square(t3 - mu) / (2 * sigmasq))) * np.cos(t1 / 2)
# def L(t1, t2, t3, sigmasq=0.04, mu=0.68):
#     sigma = np.sqrt(sigmasq)
#     return 3 * (np.cos(t1) * np.cos(t2) + (1 / (sigma * sq2pi)) * np.exp(
#         - (np.square(t3 - mu) / (2 * sigmasq))) * np.cos(t1 / 2) + 3)


def LLH(t1, t2, t3, sigmasq=0.04, mu=0.68):
    sigma = np.sqrt(sigmasq)
    return np.log(np.cos(t1) * np.cos(t2) + (1 / (sigma * sq2pi)) * np.exp(
        - (np.square(t3 - mu) / (2 * sigmasq))) * np.cos(t1 / 2)+3)

def LH_target_3d(ts):
    return (LLH(*ts))

def negLLH(t1, t2, t3,):
    sigmasq = 0.04
    mu = 0.68
    sigma = np.sqrt(sigmasq)
    return -1 * np.log(np.cos(t1) * np.cos(t2) + (1 / (sigma * sq2pi)) * np.exp(
        - (np.square(t3 - mu) / (2 * sigmasq))) * np.cos(t1 / 2)+3)


def ptrans(x):
    return x*tw


# make raster scans over all 2d combinations
def make_target_t1t2(t3_fix):
    def LH_target(ts):
        return(LLH(*ts, t3_fix))
    return LH_target

def make_target_t1t3(t2_fix):
    def LH_target(ts):
        return(LLH(ts[0], tt2_fix, t[1]))
    return LH_target

def make_target_t2t3(t1_fix):
    def LH_target(ts):
        return(LLH(t1_fix, *ts))
    return LH_target

min = Minuit(negLLH,print_level=0, t1=9, t2=3.6, t3=2.86,pedantic=False)

min.migrad(ncall=100000)

print(min.args)