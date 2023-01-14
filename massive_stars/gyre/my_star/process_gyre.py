# Import modules

import pygyre as pg
import matplotlib.pyplot as plt
import numpy as np

# Read data from a GYRE summary file

s = pg.read_output('./gyre_output/summary.h5')

# Inspect the data

print(s)

for ell in [1,]:
    for m in [0,]:
        for n in range(30):

            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)

            filename = './gyre_output/mode_ell{:03d}_m+{:02d}_n{:06d}.h5'.format(ell, m, -(n+1))
            d = pg.read_output(filename)
            print(d)

            plt.axes(ax1)
            plt.plot(d['x_ref'], d['xi_r_ref'], label='xi_r_ref')
            plt.plot(d['x_ref'], d['xi_h_ref'], label='xi_h_ref')
            plt.xlabel('x_ref')
            plt.legend()

            ell = d.meta['l']
            omega = d.meta['omega']

            x = d['x']
            V = d['V_2']*d['x']**2
            As = d['As']
            c_1 = d['c_1']
            Gamma_1 = d['Gamma_1']

            d['N2'] = d['As']/d['c_1']
            d['Sl2'] = ell*(ell+1)*Gamma_1/(V*c_1)

            plt.axes(ax2)
            plt.plot(d['x'], d['N2'], label=r'$N^2$')
            plt.plot(d['x'], d['Sl2'], label=r'S_\ell^2$')
            plt.axhline(omega.real**2, dashes=(4,2))
            plt.xlabel('x')
            plt.ylabel(r'$\omega^2$')
            plt.yscale('log')
            plt.show()

            plt.close(fig)
