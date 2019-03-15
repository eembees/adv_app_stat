import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

plt.xkcd()

dir = Path.cwd()
fname_data = list(dir.glob('*Problem2.txt'))[0]
data = np.loadtxt(fname_data)

# make polygon
data_tups = [(cor[0], cor[1]) for cor in data]
bat = Polygon(data_tups)

# make square with area 1

max_coors = (2,0.5)

num_gen   = 10000


counter  = 0

pt_list = []
arr_c   = []

for i in range(num_gen):
    coor  = ( np.random.uniform(high=max_coors[0]),  np.random.uniform(high=max_coors[1]) )
    pt = Point(coor)

    hitmiss = bat.contains(pt)

    if hitmiss:
        counter += 1

    pt_list.append(pt)
    arr_c.append(hitmiss)

arr_noc = np.ones_like(arr_c)
arr_noc[arr_c] = 0


area = counter / num_gen

x_c, y_c =  np.array([pt.coords[0] for pt in pt_list]).T

# here is the figures
fig, ax = plt.subplots()

ax.plot(data[:,0], data[:,1], linewidth=1, linestyle='dashed', label = 'outline')
# ax.plot(x_c[arr_c], y_c[arr_c], marker = '+',. linestyle = label = 'points inside' )
ax.scatter(x_c[arr_c], y_c[arr_c], marker = '+', color = 'xkcd:hot pink', label = 'inside')
ax.scatter(x_c[arr_noc], y_c[arr_noc], marker = '+', color = 'xkcd:grey', label = 'outside')

ax.set_title('Area of arbitrary polygon\n area = {}/{} = {}'.format(counter, num_gen, area))

ax.legend()

# plt.show()
fig.savefig('p2_xkcd.pdf')


