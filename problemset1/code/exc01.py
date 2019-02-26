# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-11T13:22:37+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: exc1.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T22:13:25+01:00
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('_classic_test')
plt.style.use('seaborn-dark')

# Import data
data_file = '../data/basket_data_2014.pkl'

# Make a plot
def make_hist(df, ax, data_col='AdjD', sort_col = 'Conf', names_to_use=None, range=None):
    # Start by making range components
    if range == None:
        # print(type(df[data_col].astype(np.float64)[0]))
        arr = df[data_col].astype(np.float64)
        range = ( int(arr.min() - 1), int(arr.max() + 1) )

    df.sort_values(by=sort_col, axis=0, inplace=True)
    df.set_index(keys=[sort_col], drop=False, inplace=True)

    names = df[sort_col].unique().tolist() # not really necessary but useful

    if names_to_use == None:
        names_to_use = names

    # Get series out for all the different ranges
    # For all different names
    for name in names_to_use:
        data_series = df[df[sort_col] == name][data_col].astype(np.float64)

        # Make hist
        counts, bin_edges = np.histogram(
            data_series,
            bins=10,
            range=range,
            )

        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

        s_counts = np.sqrt(counts)

        x = bin_centers[counts>0]
        y = counts[counts>0]
        # sy = s_counts[counts>0]

        ax.errorbar(x, y,
            # yerr=sy,
            xerr=1.5,
            label=name,
            fmt=',',#  ecolor='k',
            elinewidth=1, capsize=1, capthick=1)
        ax.set_xlabel(data_col)
        ax.set_ylabel('Counts')


    ax.legend()
    ax.set_title('AdjD 2014')

    # ax = df.hist( column = data_col,
    #             # by=sort_col,
    #             ax=ax )


    # for name_to_use in names_to_use:


    # print(names)

    #

    pass

if __name__ == '__main__':
    # Define start params
    names_to_use = [
    'ACC',
    'SEC',
    'B10',
    'BSky',
    'A10',
    ]

    df = pd.read_pickle(data_file)

    ## make empty plot
    # with plt.xkcd():
    fig, ax = plt.subplots()

    make_hist(df, ax, data_col='AdjD', sort_col='Conf',names_to_use=names_to_use)

    fig.tight_layout()
    fig.savefig('../plots/Exc1.pdf')
