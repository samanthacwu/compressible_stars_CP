"""
This script plots nice 3D, fancy plots of ball-shell stars.

Usage:
    ballShell_plot_volume_visual.py <root_dir> [options]

Options:
    --data_dir=<dir>     Name of data handler directory [default: slices]
    --scale=<s>          resolution scale factor [default: 1]
    --start_file=<n>     start file number [default: 1]
"""
from collections import OrderedDict

import h5py
import numpy as np
from docopt import docopt
args = docopt(__doc__)

from plotpal.file_reader import SingleTypeReader, match_basis

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

#TODO: Add outer shell, make things prettier!
def build_s2_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return phi_vert, theta_vert


def build_spherical_vertices(phi, theta, r, Ri, Ro):
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2
    r_vert = np.concatenate([[Ri], r_mid, [Ro]])
    return phi_vert, theta_vert, r_vert


def spherical_to_cartesian(phi, theta, r, mesh=True):
    if mesh:
        phi, theta, r = np.meshgrid(phi, theta, r, indexing='ij')
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

fig_name='volume_vizualization'
plotter = SingleTypeReader(root_dir, data_dir, fig_name, start_file=int(args['--start_file']), n_files=np.inf, distribution='even-write')
if not plotter.idle:
    phi_keys = []
    theta_keys = []
    with h5py.File(plotter.files[0], 'r') as f:
        scale_keys = f['scales'].keys()
        for k in scale_keys:
            if k[0] == 'phi':
                phi_keys.append(k)
            if k[0] == 'theta':
                theta_keys.append(k)
#    r_max = 1
#    shell_field = 'shell(s1_B,r=1)'
#    r_max = None
#    shell_field = 'shell(s1_S2,r=R)'
    r_max = 3.38*0.75
    shell_field = 'shell(s1_S1,r=0.75R)'
    phi_vals = ['0', '1.5707963267948966', '3.141592653589793', '4.71238898038469']
    fields = ['equator(s1_B)', 'equator(s1_S1)', 'equator(s1_S2)', shell_field] \
            + ['meridian(s1_B,phi={})'.format(phi) for phi in phi_vals] \
            + ['meridian(s1_S1,phi={})'.format(phi) for phi in phi_vals] \
            + ['meridian(s1_S2,phi={})'.format(phi) for phi in phi_vals] 
    bases  = ['r', 'phi', 'theta']


    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 0.9],projection='3d')
    cax = fig.add_axes([0.05, 0.95, 0.9, 0.04])

#    line_ax1 = fig.add_axes([0.18,0.04,0.3, 0.08])
#    line_ax2 = fig.add_axes([0.68,0.04,0.3, 0.08])

    s1_mer_data = OrderedDict()
    s1_shell_data = OrderedDict()
    s1_eq_data = OrderedDict()

    equator_line = OrderedDict()
    near_equator_line = OrderedDict()
    mer_line = OrderedDict()
    near_mer_line = OrderedDict()
    lines = [equator_line, near_equator_line, mer_line, near_mer_line]

    colorbar_dict=dict(lenmode='fraction', len=0.75, thickness=20)
    r_arrays = []

    phi_s = float(phi_vals[1])
    phi_e = float(phi_vals[3])
    theta_eq = float(np.pi/2)


    bs = {}
    first = True
    while plotter.writes_remain():
        dsets, ni = plotter.get_dsets(fields)
        time_data = plotter.current_file_handle['scales']
        theta = match_basis(dsets[shell_field], 'theta')
        phi   = match_basis(dsets[shell_field], 'phi')
        phi_de   = match_basis(dsets['equator(s1_B)'], 'phi')
        theta_de = match_basis(dsets['meridian(s1_B,phi=0)'], 'theta')
        rB_de    = match_basis(dsets['meridian(s1_B,phi=0)'], 'r')
        rS1_de   = match_basis(dsets['meridian(s1_S1,phi=0)'], 'r')
        rS2_de   = match_basis(dsets['meridian(s1_S2,phi=0)'], 'r')
        r_de = r_de_orig = np.concatenate((rB_de, rS1_de, rS2_de), axis=-1)
        dphi = phi[1] - phi[0]
        dphi_de = phi_de[1] - phi_de[0]

        if r_max is None:
            r_outer = r_de.max()
        else:
            r_outer = r_max
            r_de = r_de[r_de <= r_max]

        phi_vert, theta_vert, r_vert = build_spherical_vertices(phi, theta, r_de, 0, r_outer)
        phi_vert_de, theta_vert_de, r_vert_de = build_spherical_vertices(phi_de, theta_de, r_de, 0, r_outer)

        shell_frac = 0.9
        xo, yo, zo = spherical_to_cartesian(phi_vert, theta_vert, [shell_frac*r_outer])[:,:,:,0]
        xeq, yeq, zeq = spherical_to_cartesian(phi_vert, [theta_eq], r_vert_de)[:,:,0,:]
        xs, ys, zs = spherical_to_cartesian([phi_s], theta_vert_de, r_vert_de)[:,0,:,:]
        xe, ye, ze = spherical_to_cartesian([phi_e], theta_vert_de, r_vert_de)[:,0,:,:]

        s1_mer_data['x'] = np.concatenate([xe.T[::-1], xs.T[1:]], axis=0)
        s1_mer_data['y'] = np.concatenate([ye.T[::-1], ys.T[1:]], axis=0)
        s1_mer_data['z'] = np.concatenate([ze.T[::-1], zs.T[1:]], axis=0)

        s1_shell_data['x'] = xo
        s1_shell_data['y'] = yo
        s1_shell_data['z'] = zo

        #aim for a cutout where x > 0.
        s1_eq_data['x'] = xeq
        s1_eq_data['y'] = yeq
        s1_eq_data['z'] = zeq
 
        #Get mean properties as f(radius) // Equatorial data
        mean_s1_B  = np.expand_dims(np.mean(dsets['equator(s1_B)'][ni], axis=0), axis=0)
        mean_s1_S1 = np.expand_dims(np.mean(dsets['equator(s1_S1)'][ni], axis=0), axis=0)
        mean_s1_S2 = np.expand_dims(np.mean(dsets['equator(s1_S2)'][ni], axis=0), axis=0)
        s1_eq_B  = dsets['equator(s1_B)'][ni] - mean_s1_B
        s1_eq_S1 = dsets['equator(s1_S1)'][ni] - mean_s1_S1
        s1_eq_S2 = dsets['equator(s1_S2)'][ni] - mean_s1_S2
        radial_s1_mean = np.concatenate((mean_s1_B, mean_s1_S1, mean_s1_S2), axis=-1)
        eq_field_s1 = np.concatenate((s1_eq_B, s1_eq_S1, s1_eq_S2), axis=-1)
        radial_scaling = np.sqrt(np.mean(eq_field_s1**2, axis=0))
        eq_field_s1 /= radial_scaling
        minmax_s1 = 2*np.std(eq_field_s1)
        s1_eq_data['surfacecolor'] = np.pad(eq_field_s1.squeeze()[:, r_de_orig <= r_outer], ( (1, 0), (1, 0) ), mode='edge')
        eq_nan_bool = s1_eq_data['x'] >= 0
        s1_eq_data['surfacecolor'] = np.where(eq_nan_bool, s1_eq_data['surfacecolor'], np.nan)

        print(s1_eq_data)

        #Get meridional slice data
        mer_0_s1_B  = (dsets['meridian(s1_B,phi={})'.format(phi_vals[1])][ni] - mean_s1_B).squeeze()
        mer_1_s1_B  = (dsets['meridian(s1_B,phi={})'.format(phi_vals[3])][ni] - mean_s1_B).squeeze()
        mer_0_s1_S1 = (dsets['meridian(s1_S1,phi={})'.format(phi_vals[1])][ni] - mean_s1_S1).squeeze()
        mer_1_s1_S1 = (dsets['meridian(s1_S1,phi={})'.format(phi_vals[3])][ni] - mean_s1_S1).squeeze()
        mer_0_s1_S2 = (dsets['meridian(s1_S2,phi={})'.format(phi_vals[1])][ni] - mean_s1_S2).squeeze()
        mer_1_s1_S2 = (dsets['meridian(s1_S2,phi={})'.format(phi_vals[3])][ni] - mean_s1_S2).squeeze()
        #Calculate midpoints meridionally.
        mer_0_s1_B  = (mer_0_s1_B[1:,:]  + mer_0_s1_B[:-1,:])/2
        mer_0_s1_S1 = (mer_0_s1_S1[1:,:] + mer_0_s1_S1[:-1,:])/2
        mer_0_s1_S2 = (mer_0_s1_S2[1:,:] + mer_0_s1_S2[:-1,:])/2
        mer_1_s1_B  = (mer_1_s1_B[1:,:]  + mer_1_s1_B[:-1,:])/2
        mer_1_s1_S1 = (mer_1_s1_S1[1:,:] + mer_1_s1_S1[:-1,:])/2
        mer_1_s1_S2 = (mer_1_s1_S2[1:,:] + mer_1_s1_S2[:-1,:])/2

        mer_0_s1 = np.concatenate((mer_0_s1_B, mer_0_s1_S1, mer_0_s1_S2), axis=-1)/radial_scaling
        mer_1_s1 = np.concatenate((mer_1_s1_B, mer_1_s1_S1, mer_1_s1_S2), axis=-1)/radial_scaling
        mer_0_s1 = mer_0_s1.squeeze()[:, r_de_orig <= r_outer]
        mer_1_s1 = mer_1_s1.squeeze()[:, r_de_orig <= r_outer]

        mer_s1 = np.concatenate([mer_1_s1.transpose((1,0))[::-1], mer_0_s1.transpose((1,0))], axis=0)
        mer_s1 = np.pad(mer_s1, ((1, 0), (1, 1)), mode='edge')
        s1_mer_data['surfacecolor'] = mer_s1
        mer_nan_bool = s1_mer_data['z'] >= 0
        s1_mer_data['surfacecolor'] = np.where(mer_nan_bool, s1_mer_data['surfacecolor'], np.nan)

        #Get shell slice data
        s1_S_r095R = dsets[shell_field][ni] - np.expand_dims(np.mean(np.mean(dsets[shell_field][ni], axis=2), axis=1), axis=[1,2])
        shell_s1 = s1_S_r095R.squeeze()
        shell_s1 /= np.sqrt(np.mean(shell_s1**2))
        shell_s1 = np.pad(shell_s1, ((0, 1), (1, 0)), mode='edge')
        s1_shell_data['surfacecolor'] = shell_s1
        shell_nan_bool = np.logical_or((xo < 0), (zo < 0))
        s1_shell_data['surfacecolor'] = np.where(shell_nan_bool, s1_shell_data['surfacecolor'], np.nan)

        cmap = matplotlib.cm.get_cmap('RdBu_r')
        norm = matplotlib.colors.Normalize(vmin=-minmax_s1, vmax=minmax_s1)

        data = OrderedDict()
        for k in ['x', 'y', 'z', 'surfacecolor']:
            data[k] = np.concatenate([d[k] for d in [s1_shell_data, s1_mer_data]])
        if first:
            for i, d in enumerate([data, s1_eq_data]):
#            for i, d in enumerate([s1_shell_data, s1_eq_data, s1_mer_data]):
#            for i, d in enumerate([s1_shell_data, s1_mer_data, s1_eq_data]):
#            for i, d in enumerate([s1_mer_data, s1_shell_data, s1_eq_data]):
                print('plotting data {}'.format(i))
                x = d['x']
                y = d['y']
                z = d['z']
                sfc = cmap(norm(d['surfacecolor']))
                surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
                d['surf'] = surf
                #wireframe doesn't seem to be working
#                ax.plot_wireframe(x, y, z, ccount=1, rcount=1, linewidth=1, color='black')
            ax.set_xlim(-0.7*r_outer, 0.7*r_outer)
            ax.set_ylim(-0.7*r_outer, 0.7*r_outer)
            ax.set_zlim(-0.7*r_outer, 0.7*r_outer)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(azim=-25, elev=25)
#            ax.axis('off')
            first = False
        else:
            for i, d in enumerate([s1_shell_data, s1_mer_data]):
                sfc = cmap(norm(d['surfacecolor']))
                d['surf'].set_facecolors(sfc.reshape(sfc.size//4,4))
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', format=FormatStrFormatter('%.1f'))
        cbar.set_label('normalized s1' + ', t = {:.3e}'.format(time_data['sim_time'][ni]))
#        cbar.set_ticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
        #TODO: fix colorbar tick flickering when plotting in parallel

#        line1 = line_ax1.plot(r_de.ravel(), radial_s1_mean.ravel())
#        line2 = line_ax2.semilogy(r_de.ravel(), radial_scaling.ravel())
#        line_ax1.set_ylabel(r'$s_1(r)$')
#        line_ax2.set_ylabel(r'$\sigma(s_1)(r)$')

        fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, time_data['write_number'][ni]), dpi=200, bbox_inches='tight')
#        for pax in [cax, line_ax1, line_ax2]:
        for pax in [cax, ]:
           pax.clear()
        import sys
        sys.exit()
