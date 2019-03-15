# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-12T10:21:25+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: monte_carlo.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T11:37:31+01:00
import numpy as np


def monte_carlo(pdffunc, num=1000, range_x: tuple = (0.01, 1)):
    """Short summary.

    Parameters
    ----------
    pdffunc : function
        PDF function, used in accepting or rejecting points
        Takes one x and returns output
    num : int
        number of points to generate
    range_x : tuple
        Range in which to generate x values, and corresponding y values

    Returns
    -------
    output_arr : ndarray
        Array of size (num), with monte carlo generated values

    """
    output_arr = np.zeros(num)
    i = 0

    x_test_arr = np.linspace(*range_x, num=100)
    try:
        y_test_arr = pdffunc(x_test_arr)
    except ValueError:  # slower but functional method for homemade kde
        y_test_arr = np.array([pdffunc(x) for x in x_test_arr])

    range_y = (0, y_test_arr.max())

    while i < num:
        # generate a numbers
        test_x = (range_x[1] - range_x[0]) * np.random.random() + range_x[0]
        test_y = (range_y[1] - range_y[0]) * np.random.random() + range_y[0]

        test_y_cond = pdffunc(test_x)

        if test_y < test_y_cond:
            output_arr[i] = test_x
            i += 1

    return output_arr


if __name__ == '__main__':
    from plot_gaussian import plot_gaussian
    import matplotlib.pyplot as plt


    def funk(x):
        return (1 + 2 * x + 4 * x ** 2)


    data = monte_carlo(funk)

    fig, ax = plt.subplots()

    print(data)

    ax.hist(data)

    plt.show()
