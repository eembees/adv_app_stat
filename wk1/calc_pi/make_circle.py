import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from plot_gaussian import plot_gaussian


def make_random_cirle(r, n_points, line_edge, VERBOSE=False):
    ## First, make a lot of x, y points
    coors = np.random.random_sample( size = (n_points,2))


    ## Find number of points that intersect with circle edge
    ## Find number of points that intersect with x axis (length 1)

    counter_edge = 0
    counter_axis = 0
    for coor in coors:
        coor_r = np.sqrt( coor[0]**2 + coor[1]**2 )

        if ( (r - line_edge) <= coor_r <= (r + line_edge) ):
            counter_edge += 1

        if ( ( - line_edge ) <= coor[0] <= (2 * line_edge ) ):
            # 2 times line edge, because nothing is below the axis
            counter_axis += 1
    if VERBOSE:
        print('counter_edge')
        print(counter_edge)
        print('counter_axis')
        print(counter_axis)


    ## Divide to get circumference
    ## times four because it's a quarter
    C_approx = (counter_edge / counter_axis ) * 4
    if VERBOSE:
        print('C_approx')
        print(C_approx)
    ## Use circumference to calc area
    ## A = 2*r*C

    A = 2 * r * C_approx

    if VERBOSE:
        print('A')
        print(A)
    ## plotting
    # fig, ax  = plt.subplots()
    #
    # ax.scatter(x = coors[:,0], y = coors[:,1], s=2, c='xkcd:hot pink')
    # # ax.
    # plt.show()


    return A, C_approx


def generate_many_circles(r, n_points, line_edge, n_loops):
    A_arr = np.empty(shape=n_loops)
    C_arr = np.empty(shape=n_loops)

    for i in range(n_loops):
        A_temp, C_temp = make_random_cirle(r, n_points, line_edge,)
        A_arr[i]    = A_temp
        C_arr[i]    = C_temp

    return A_arr, C_arr

## running
if __name__ =='__main__':
    ## Set params
    n_points = 10**4
    line_edge= 10**-2

    r = 1
    n_loops = 100

    ## Do stuff

    A_arr, C_arr = generate_many_circles(r, n_points, line_edge, n_loops)

    # print(A_arr)
    # print(C_arr)

    ## Make a plot of values of pi
    pi_arr = C_arr / 2.

    fig, ax = plt.subplots()

    ax = plot_gaussian(pi_arr, ax)

    #
    # ax.hist(
    # pi_arr,
    # bins=100
    # )

    plt.show()
