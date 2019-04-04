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



# 3d nested sampling here
npoints_3d = 5000
res = ne.sample(LH_target_3d, prior_transform=ptrans, ndim=3, npoints = npoints_3d,
                method='multi',
                # update_interval=20,
                )
print('sampling done')


# get the ml solution out
# ts_ml = res.samples[np.unravel_index(np.argmax(res.logl, axis=None), res.logl.shape)]
# print(res.summary)
# print(ts_ml)
# exit()
figc = corner.corner(res.samples, weights=res.weights, bins = 40,labels=['t1','t2','t3'])#, range=ranges,)# plot_contours=False)
# fig = corner.corner(res.samples[res.weights > np.median(res.weights)])#, weights=res.weights, labels=['t1','t2','t3'], range=ranges, plot_contours=False)
figc.savefig('./figs/p5_nest_3d_corner_{}.png'.format(npoints_3d), dpi=300)

# plot the result from 3d sampling on t1 and t3 axes
# fig1, ax1 = plt.subplots()

g1 = sns.jointplot(x = res.samples[-100:,0],y = res.samples[-100:,2], kind='scatter', joint_kws={'marker':'1'})

g1.set_axis_labels('theta1','theta3')

g1.savefig('./figs/p5_nest_3d_scatter_t1t3_{}.pdf'.format(npoints_3d))

with open('nestle_sampler{}.pkl'.format(npoints_3d), 'wb') as writefile:
    pickle.dump(res, writefile)