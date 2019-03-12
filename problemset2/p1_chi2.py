import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from lib.nice_string_output import nice_string_output



# set pars
arr_x = np.linspace(start=0, stop=10, num=1000)

dof = 1

names = [
    'Name',
    'ID',
    'Date',
    'NDOF',
]

values = [
    'Magnus Berg Sletfjerding',
    'dzl123',
    '12 March 2019',
    '1'
]


string_text = nice_string_output(names, values, extra_spacing=2)


## QUESTION
# is it chisquared that should be distributed 0 to 10 or x_
# TODO ask Jason about thsi



pdf = chi2.pdf(arr_x, dof)

fig, ax = plt.subplots()

ax.plot(pdf, arr_x)

ax.set_xlabel('$\chi^2$')

ax.set_ylabel('Probability density')

ax.text(1.95, 8.5,string_text, family='monospace')

fig.tight_layout()

fig.savefig('p1_chi2.pdf')