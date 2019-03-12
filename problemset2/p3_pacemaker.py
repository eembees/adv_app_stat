import numpy as np
import pandas as pd
from io import StringIO

def find_relative_defective(df: pd.DataFrame):
    df = df.copy(deep=True)
    df['perc_def'] = df.production * df.defective
    p_d = np.sum(df.defective.values * df.production.values)

    defective_r = []
    for row_i in df.index:
        row = df.loc[row_i]
        pxd = row.defective * row.production / p_d
        defective_r.append(pxd)

    df['prob_defective'] = defective_r

    return df


def find_new_defective_rates(df):
    df = df.copy(deep=True)

    # determine which row to normalize by
    rates_candidates = []

    # test a series of ks

    defective_rates_new_nonorm = 1 / (df.production.values)


    for row_i in df.index:
        row = df.loc[row_i]
        # Calculate all the new defective rates
        defective_rates_row_norm = defective_rates_new_nonorm * (row.defective / defective_rates_new_nonorm[row_i])
        # Only if all rates are higher or equal, accept the candidate
        if all(defective_rates_row_norm >= df.defective.values):
            rates_candidates.append(defective_rates_row_norm)


    if len(rates_candidates) == 1: # only one candidate is accepted
        df['new_defective'] = rates_candidates[0]
    else: # find the minimization
        overall_defective_rates = [
            np.sum(rates_candidate * df.production.values) for rates_candidate in rates_candidates
        ]

        rate_min_ind = np.argmin(overall_defective_rates)

        df['new_defective'] = rates_candidates[rate_min_ind]

    return df

df = pd.DataFrame(
    {
        'facility': ['A1','A2','A3','A4','A5'],
        'production': np.array([35,15,5,20,25])/100,
        'defective': np.array([2,4,10,3.5,3.1])/100,
    }
)

# q1
# if defective, what is the prb that it came from A2?

df = find_relative_defective(df)

d_a2 = df.loc[df.facility == 'A2']

d_max = df.loc[df.prob_defective.values.argmax()]
print('Problem 3a:')
print('Current aggregate defective rate: {}'.format(
    np.sum(df.production.values * df.defective.values)))
print('If defective, a PM is {:.2%} likely to come from the facility {}'.format(
    d_a2.prob_defective.values[0], d_a2.facility.values[0]))
print('If defective, a PM is {:.2%} likely to come from the facility {}'.format(
    d_max.prob_defective, d_max.facility))

## q2
## find how to adjust defective rates for all facilities to minimize total defective rate


# print(df)
#
df = find_new_defective_rates(df)
print(df[['facility', 'defective', 'new_defective']])



data2 = StringIO('''
facility     production        defective
A1           0.27              0.02
A2           0.1               0.04
A3           0.05              0.1
A4           0.08              0.035
A5           0.25              0.022
A6           0.033             0.092
A7           0.019             0.12
A8           0.085             0.07
A9           0.033             0.11
A10          0.02              0.02
A11          0.015             0.07
A12          0.022             0.06
A13          0.015             0.099
A14          0.008             0.082
''')

df2 = pd.read_csv(data2, delim_whitespace=True, usecols=['facility','production','defective'])


# print(df2)

df2 = find_relative_defective(df2)
df2 = find_new_defective_rates(df2)

print(df2[['facility', 'defective', 'new_defective']])