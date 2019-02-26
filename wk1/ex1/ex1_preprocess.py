# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-05T10:44:38+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: ex1_preprocess.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T11:55:55+01:00



from glob import glob
import pandas as pd
import numpy as np


# find file
frankfile = glob("*Numbers.txt")[0]

print(frankfile)

f = open(frankfile, 'r')

# print(type(f.read()))

datasets = f.read().split("Data set ")

for dataset in datasets[1:]:



# for line in f:
    # if line[0] != 'M':
    #     if line[0:4] == "Data" :
    #         fname = line
    #         tempfile = ''
    #         print("start new line")
    #     tempfile += (line+'\n')
    #     if line == '\n':
    #         print("hello")
    #         print(type(tempfile))
    # else:
    #     print("Start of the file")
