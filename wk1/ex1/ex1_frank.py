# Frank's numbers
## Importing packages
from glob import glob

import numpy as np
import pandas as pd


## Import file
path = glob('frank*.txt')

df = pd.read_csv(path)

print(df.head())
