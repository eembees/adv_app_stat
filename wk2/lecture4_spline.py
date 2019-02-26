import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.integrate import quad
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

# datafile = '/Users/mag/Documents/study/msc/advanced_applied_stat/wk2/DustLog_forClass.dat.txt'
datafile = './SplineCubic.txt'

dustDf = pd.read_csv(datafile, header=None, sep=' ', names=['Depth', 'OptLog'])



# spline
xold = dustDf.Depth

xnew = np.linspace(xold.min(), xold.max(), num=10000)


def spline(xdata, ydata, kind='linear', num=10000, int_range=None):
    range_x = (xdata.min(),xdata.max())
    xnew = np.linspace(*range_x, num=num)
    if kind == 'pchip':
        f = PchipInterpolator(xdata,ydata)
    else:
        f = interp1d(xdata, ydata, kind=kind)
    if int_range is None:
        int = quad(f, *range_x)[0]
    else:
        int = quad(f, *int_range)[0]
    y = f(xnew)
    return(f,y,int)


# plot data

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4,9),sharex='all')
ax = axes.ravel()

(f_lin, y_lin, int_lin) = spline(xold, dustDf.OptLog, )
(f_qud, y_qud, int_qud) = spline(xold, dustDf.OptLog, kind='quadratic')
(f_cub, y_cub, int_cub) = spline(xold, dustDf.OptLog, kind='cubic')

ax[0].scatter(dustDf.Depth, dustDf.OptLog, c = 'r',)

ax[0].plot(xnew, y_lin, '--', label='linear    int={:.2f}'.format(int_lin))
ax[0].plot(xnew, y_qud, '--', label='quadratic int={:.2f}'.format(int_qud))
ax[0].plot(xnew, y_cub, '--', label='cubic     int={:.2f}'.format(int_cub))

(f_lin, y_lin, int_lin) = spline(xold, dustDf.OptLog, int_range=(0.03,0.1))
(f_qud, y_qud, int_qud) = spline(xold, dustDf.OptLog, int_range=(0.03,0.1), kind='quadratic')
(f_cub, y_cub, int_cub) = spline(xold, dustDf.OptLog, int_range=(0.03,0.1), kind='cubic')

ax[1].scatter(dustDf.Depth, dustDf.OptLog, c='r', )

ax[1].plot(xnew, y_lin, '--', label='linear    int={:.2f}'.format(int_lin))
ax[1].plot(xnew, y_qud, '--', label='quadratic int={:.2f}'.format(int_qud))
ax[1].plot(xnew, y_cub, '--', label='cubic     int={:.2f}'.format(int_cub))

(f_lin, y_lin, int_lin) = spline(xold, dustDf.OptLog, int_range=(0.03,0.1), kind = 'pchip')

ax[2].scatter(dustDf.Depth, dustDf.OptLog, c='r', )

ax[2].plot(xnew, y_lin, '--', label='Pchi   p    int={:.2f}'.format(int_lin))


ax[0].set_xlim(0.0,0.35)



#
# ax.set_xlim(0, 0.1)
# ax.set_xlim(2080, 2100)
# ax.set_ylim(100,200)


ax[0].legend()
ax[1].legend()
ax[2].legend()


fig.tight_layout()
plt.show()

