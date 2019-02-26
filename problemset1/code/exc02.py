# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-11T13:22:37+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: exc1.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T15:38:15+01:00
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_gaussian import *
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

    colors = [
    'xkcd:pastel blue',
    'xkcd:pastel yellow',
    'xkcd:pastel purple',
    'xkcd:pastel pink',
    'xkcd:pastel red',
    'xkcd:pastel orange',
    ]

    data_col = 'AdjO'

    df_14 = pd.read_pickle(data_file)
    df_09 = pd.read_pickle(data_file.replace('2014', '2009'))

    # print(df_14.columns)
    # exit()

    names = df_14.Conf.unique().tolist() + df_09.Conf.unique().tolist() # not really necessary but useful


    # sort both by team name
    df_14.sort_values(by='Team', axis=0, inplace=True)
    df_14.set_index(keys=['Team'], drop=False,inplace=True)

    df_09.sort_values(by='Team', axis=0, inplace=True)
    df_09.set_index(keys=['Team'], drop=False,inplace=True)

    # Append new series to df
    AdjO_14 = df_14.AdjO.astype(np.float64)
    AdjO_09 = df_09.AdjO.astype(np.float64)

    AdjO_diff = AdjO_14 - AdjO_09


    df_AdjO = AdjO_09.to_frame()
    df_AdjO['AdjO_diff'] = AdjO_diff
    df_AdjO['Conf'] = df_09.Conf
    # print(df_AdjO)
    ## make empty plot

    fig, ax = plt.subplots()

    adjo_md_vals = ['AdjO diff-Md',]
    adjo_mn_vals = ['AdjO diff-Mn',]
    adjo_md_names= ['Conference',]


    for name, color in zip(names_to_use, colors):
        # print("Now presenting")
        # print(name)
        # print(color)

        data_series = df_AdjO[df_AdjO['Conf'] == name]#.astype(np.float64)

        # df_AdjO.plot.scatter(
        # x = 'AdjO',
        # y = 'AdjO_diff',
        # c = color,
        # label = name,
        # ax = ax,
        # )

        # print(data_series.AdjO.shape)
        # print(data_series.AdjO_diff.shape)

        ax.scatter(
        x = data_series.AdjO,
        y = data_series.AdjO_diff,
        c = color,
        marker = '1',
        label = name,
        )


        ax.set_xlabel('AdjO 2009')
        ax.set_ylabel('AdjO 2014 - AdjO 2009')

        # here we make a string with the median
        median_adjo = data_series.AdjO_diff.median()
        adjo_md_vals.append('{:.2f}'.format(median_adjo))
        adjo_mn_vals.append('{:.2f}'.format(data_series.AdjO_diff.mean()))
        adjo_md_names.append(name)

    # Find median of not-included teams
    loser_names = list(set(names) - set(names_to_use))
    series_losers = df_AdjO.copy()
    for winner_name in names_to_use:
        series_losers.drop(series_losers.index[series_losers.Conf == winner_name], inplace=True)

    adjo_md_names.append("Outsiders")
    adjo_md_vals.append('{:.2f}'.format(series_losers.AdjO_diff.median()) )
    adjo_mn_vals.append('{:.2f}'.format(series_losers.AdjO_diff.mean()) )

    # print(adjo_md_names, adjo_md_vals)

    md_string = nice_string_output(adjo_md_names, adjo_md_vals, extra_spacing=2)
    mn_string = nice_string_output(adjo_md_names, adjo_mn_vals, extra_spacing=2)

    text_file = open('../plots/exc02_output.txt','w')
    text_file.write(md_string)
    text_file.write('\n')
    text_file.write(mn_string)
    text_file.close()

    ax.legend()
    ax.set_title('AdjD 2014 - AdjD 2009')

    fig.tight_layout()
    fig.savefig('../plots/Exc2.pdf')
