import numpy as np
import math

## FROM
# https://stackoverflow.com/a/20926435/9014887

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def step_func(x):
    return 1 if x >= 0 else 0

#
# def get_angle(pt1, pt2):
#     return np.arccos(np.dot(pt1, pt2))


def within_phi(pt1, pt2, cos_phi):
    # angle = get_angle(pt1, pt2)
    #
    # cos_pts = np.cos(angle)
    cos_pts = np.dot(pt1, pt2)

    cos_diff = cos_pts - cos_phi

    return(step_func(cos_diff))

def count_pair_events(pts_all, phi):
    counter = 0
    nTot = len(pts_all)
    cos_phi = np.cos(phi)
    for i in range(len(pts_all)):
        for j in range(i+1, len(pts_all)):
            # if i != j:
            counter += within_phi(pts_all[i],pts_all[j],cos_phi)

    # counter = 2/(nTot*(nTot - 1)) * counter

    return counter


# generate some events
nEvents = 1000
phi = np.radians(20)
angles = np.random.uniform(low = 0, high = 2* np.pi, size = (nEvents, 2))

xyzs   = [sph2cart(ang[0], ang[1], 1) for ang in angles  ]


print(count_pair_events(xyzs, phi))


