import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.lib_math as lm
import lib.nice_string_output as ns
import corner

import seaborn as sns
import nestle as ne
import pickle


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

with open('nestle_sampler5000.pkl', 'rb') as writefile:
    res = pickle.load(writefile)



# find most likely solutions (1000) and get them out

num_ml = 12383
inds_max = np.argpartition(np.abs(res.logl), -1 * num_ml)[-1 * num_ml:]

fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(res.samples[inds_max, 0], res.samples[inds_max, 2], marker = '2',s=1)
ax.set_title('Nested sampling of {}/12384 highest LLH values '.format(num_ml))
ax.set_xlim(t1_range)
ax.set_ylim(t3_range)

ax.set_xlabel(r'$\theta 1$')
ax.set_ylabel(r'$\theta 3$')

fig.tight_layout()

fig.savefig('./figs/p5_nest_5000_max{}.pdf'.format(num_ml))

fig, axes = plt.subplots(figsize=(7,7),nrows=2,ncols=2)

axs = axes.ravel()

# # print(res.summary())
# ns = [1000,100,10,3]
# for ni, num_ml in enumerate(ns):
#     inds_max = np.argpartition(np.abs(res.logl), -1 * num_ml)[-1 * num_ml:]
#
#     axs[ni].scatter(res.samples[inds_max,0], res.samples[inds_max, 2],marker = '2')
#     axs[ni].set_title('{}/12384 LLH values '.format(num_ml))
#
#     axs[ni].set_xlim(t1_range)
#     axs[ni].set_ylim(t3_range)
#
#     axs[ni].set_xlabel(r'$\theta 1$')
#     axs[ni].set_ylabel(r'$\theta 3$')
#
# fig.tight_layout()
#
# fig.savefig('./figs/p5_nest_5000_grid.pdf')