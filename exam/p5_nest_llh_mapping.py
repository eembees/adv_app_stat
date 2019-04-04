import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.lib_math as lm
import lib.nice_string_output as ns
import corner

import seaborn as sns
import nestle as ne
import pickle
from matplotlib.lines import Line2D


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
def raster_scan(range_a, range_b,  pdf, ax, dims = 50, plot_llh_contours = True):
    """
    Makes 2d raster scan for pdf in the specified ranges
    :param data:
    :param range_a:
    :param range_b:
    :param pdf: callable, must be of the forma f(x,a,b) for llh estimation
    :param ax:
    :param dims:
    :param plot_llh_contours:
    :return:
    """
    vals_a   = np.linspace(*range_a, num=dims)
    vals_b   = np.linspace(*range_b, num=dims)

    a, b = np.meshgrid(vals_a, vals_b, indexing='ij')

    a_3d = a[np.newaxis, :, :]
    b_3d = b[np.newaxis, :, :]

    rst_llh = pdf(a_3d, b_3d,)[0]

    ax.pcolormesh(a, b, rst_llh)

    rst_max_idx = np.nanargmax(rst_llh)
    a_idx, b_idx = np.unravel_index(rst_max_idx, rst_llh.shape)
    rst_max_b = vals_a[a_idx]
    rst_max_a = vals_b[b_idx]
    rst_max_llh = rst_llh[a_idx, b_idx]

    if plot_llh_contours == True:
        llh_1s = rst_max_llh - 0.5
        llh_2s = rst_max_llh - 2.
        llh_3s = rst_max_llh - 4.5

        llhs_s = [
            llh_1s,
            llh_2s,
            llh_3s,
        ]

        cf_colors = [
            'xkcd:dark purple',
            'xkcd:purple',
            'xkcd:light purple',
        ]


        ax.contour(a, b, rst_llh, levels=sorted(llhs_s),colors = cf_colors, label='CF')

        legend_elements = [Line2D([0], [0], color=cf_colors[0], label='LLH 3s'),
                           Line2D([0], [0], color=cf_colors[1], label='LLH 2s'),
                           Line2D([0], [0], color=cf_colors[2], label='LLH 1s'),]


        ax.legend(handles=legend_elements)


    return ax


# make raster scans over all 2d combinations
def make_target_t1t2(t3_fix):
    def LH_target(t1,t2):
        return(LLH(t1,t2, t3_fix))
    return LH_target

def make_target_t1t3(t2_fix):
    def LH_target(t1,t3):
        return(LLH(t1, t2_fix, t3))
    return LH_target

def make_target_t2t3(t1_fix):
    def LH_target(t2,t3):
        return(LLH(t1_fix, t2,t3))
    return LH_target

with open('nestle_sampler5000.pkl', 'rb') as writefile:
    res = pickle.load(writefile)

# now RES is our file that we use for


ts_ml = res.samples[np.unravel_index(np.argmax(res.logl, axis=None), res.logl.shape)]


fig_map, ax_m = plt.subplots(ncols=3, figsize = (10,4) )

# 2d raster scans now
ax_m[0] = raster_scan(t1_range, t2_range, make_target_t1t2(ts_ml[2]), ax_m[0])
ax_m[0].set_xlabel(r'$\theta 1$')
ax_m[0].set_ylabel(r'$\theta 2$')


ax_m[1] = raster_scan(t1_range, t3_range, make_target_t1t3(ts_ml[1]), ax_m[1])
ax_m[1].set_xlabel(r'$\theta 1$')
ax_m[1].set_ylabel(r'$\theta 3$')

ax_m[2] = raster_scan(t2_range, t3_range, make_target_t2t3(ts_ml[0]), ax_m[2])
ax_m[2].set_xlabel(r'$\theta 2$')
ax_m[2].set_ylabel(r'$\theta 3$')





fig_map.tight_layout()
fig_map.savefig('./figs/p5_nest_3llhs_rasters.pdf')