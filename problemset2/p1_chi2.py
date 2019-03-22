import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

from scipy.stats import chi2
from lib.nice_string_output import nice_string_output

# plt.xkcd()

# set pars
arr_x = np.linspace(start=0, stop=10, num=1000)

dof = 1

names = [
    'Name',
    'ID',
    'Date',
    'NDOF',
    'Course',
    '  ',
    'Title',
]

values = [
    'Magnus Berg Sletfjerding',
    'dzl123',
    '12 March 2019',
    '1',
    'Advanced Methods in',
    'Applied Statistics',
    'Problem Set 2'
]

string_text = nice_string_output(names, values, extra_spacing=2)

## QUESTION
# is it chisquared that should be distributed 0 to 10 or x_
# TODO ask Jason about thsi


pdf = chi2.pdf(arr_x, dof)

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(arr_x,pdf, color='xkcd:forest green', linewidth=5)

ax.set_xlabel('$\chi^2$')

ax.set_ylabel('Probability density')

ax.text(0.42, 0.45, string_text, fontsize=20, family='monospace', transform=ax.transAxes)

ax.set_yscale('log')
# Add watermark
im = image.imread('ku_logo.png')
fig.figimage(im, 10, 10, zorder=3, resize=True, alpha=0.3)

# fig.tight_layout()

# ax.set_xticklabels(ax.get_ticklabels, fontsize=15)
# ax.set_yticklabels(yticklabels, fontsize=15)

ax.tick_params(axis='both', labelsize=20)
ax.xaxis.label.set_size(25)
ax.yaxis.label.set_size(25)
# ax.set_title('Advanced Methods in Applied Statistics - Problem Set 2')
# fig.savefig('p1_chi2_xkcd.pdf')
fig.savefig('p1_chi2.pdf')
