"""
This script plots nice 3D, fancy plots of ball-shell stars.

Usage:
    ballShell_plot_volume_visual.py <root_dir> [options]

Options:
    --data_dir=<dir>     Name of data handler directory [default: slices]
    --scale=<s>          resolution scale factor [default: 1]
    --start_file=<n>     start file number [default: 1]
"""
import gc
from collections import OrderedDict

from mpi4py import MPI
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

    dummy_fig = plt.figure(figsize=(4,4))
    dummy_ax  = dummy_fig.add_axes([0, 0, 1, 0.9],projection='3d')
    dummy_cax = dummy_fig.add_axes([0.05, 0.95, 0.9, 0.04])

#    line_ax1 = fig.add_axes([0.18,0.04,0.3, 0.08])
#    line_ax2 = fig.add_axes([0.68,0.04,0.3, 0.08])

    s1_mer_data = OrderedDict()
    s1_shell1_data = OrderedDict()
    s1_shell2_data = OrderedDict()
    s1_eq_data = OrderedDict()

    equator_line = OrderedDict()
    near_equator_line = OrderedDict()
    mer_line = OrderedDict()
    near_mer_line = OrderedDict()
    lines = [equator_line, near_equator_line, mer_line, near_mer_line]

    colorbar_dict=dict(lenmode='fraction', len=0.75, thickness=20)
    r_arrays = []

    phi_mer1 = float(phi_vals[2])
    theta_eq = float(np.pi/2)


    bs = {}
    first = True
    while plotter.writes_remain():
        dsets, ni = plotter.get_dsets(fields)
        time_data = plotter.current_file_handle['scales']
        print('got time data')


        #Only get grid info on first pass
        if first:
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

            shell_frac = 1 
            shell1_theta_pick = theta_vert >= np.pi/2
            shell2_theta_pick = np.isfinite(theta_vert)
            shell1_theta_pick_field = theta >= np.pi/2
            shell2_theta_pick_field = np.isfinite(theta)
            shell1_phi_pick = phi_vert <= np.pi*1.05
            shell2_phi_pick = phi_vert > np.pi 
            shell1_phi_pick_field = phi <= np.pi*1.05
            shell2_phi_pick_field = phi > np.pi 
            eq_phi_pick = phi_vert_de <= np.pi
            xo, yo, zo = spherical_to_cartesian(phi_vert[shell1_phi_pick], theta_vert[shell1_theta_pick], [shell_frac*r_outer])[:,:,:,0]
            xo2, yo2, zo2 = spherical_to_cartesian(phi_vert[shell2_phi_pick], theta_vert[shell2_theta_pick], [shell_frac*r_outer])[:,:,:,0]
            xeq, yeq, zeq = spherical_to_cartesian(phi_vert_de[eq_phi_pick], [theta_eq], r_vert_de)[:,:,0,:]


            theta_mer = np.concatenate([-theta_de, theta_de[::-1]])
            mer_theta_pick = np.abs(theta_mer) <= np.pi
            mer_theta_pick = np.abs(theta_mer) <= np.pi
            x_mer, y_mer, z_mer = spherical_to_cartesian([phi_mer1,], theta_mer[mer_theta_pick], r_vert_de)[:,0,:,:]

            s1_mer_data['x'] = x_mer
            s1_mer_data['y'] = y_mer
            s1_mer_data['z'] = z_mer

            s1_shell1_data['x'] = xo
            s1_shell1_data['y'] = yo
            s1_shell1_data['z'] = zo
            s1_shell2_data['x'] = xo2
            s1_shell2_data['y'] = yo2
            s1_shell2_data['z'] = zo2

            #aim for a cutout where x > 0.
            s1_eq_data['x'] = xeq
            s1_eq_data['y'] = yeq
            s1_eq_data['z'] = zeq
        print('past if statement')

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
        s1_eq_data['surfacecolor'] = np.pad(eq_field_s1.squeeze()[:, r_de_orig <= r_outer], ( (1, 0), (1, 0) ), mode='edge')[eq_phi_pick,:]
        print('past equator')


        #Get meridional slice data
        mer_0_s1_B  = (dsets['meridian(s1_B,phi={})'.format(phi_vals[0])][ni] - mean_s1_B).squeeze()
        mer_1_s1_B  = (dsets['meridian(s1_B,phi={})'.format(phi_vals[2])][ni] - mean_s1_B).squeeze()
        mer_0_s1_S1 = (dsets['meridian(s1_S1,phi={})'.format(phi_vals[0])][ni] - mean_s1_S1).squeeze()
        mer_1_s1_S1 = (dsets['meridian(s1_S1,phi={})'.format(phi_vals[2])][ni] - mean_s1_S1).squeeze()
        mer_0_s1_S2 = (dsets['meridian(s1_S2,phi={})'.format(phi_vals[0])][ni] - mean_s1_S2).squeeze()
        mer_1_s1_S2 = (dsets['meridian(s1_S2,phi={})'.format(phi_vals[2])][ni] - mean_s1_S2).squeeze()
        #Calculate midpoints meridionally.

        mer_0_s1 = np.concatenate((mer_0_s1_B, mer_0_s1_S1, mer_0_s1_S2), axis=-1)/radial_scaling
        mer_1_s1 = np.concatenate((mer_1_s1_B, mer_1_s1_S1, mer_1_s1_S2), axis=-1)/radial_scaling
        mer_0_s1 = mer_0_s1.squeeze()[:, r_de_orig <= r_outer]
        mer_1_s1 = mer_1_s1.squeeze()[:, r_de_orig <= r_outer]


        #go from theta = pi -> 0 on RHS slice, then 0 -> pi on LHS slice.
        mer_s1 = np.concatenate( (mer_0_s1, mer_1_s1[::-1,:]), axis=0)
        mer_s1 = np.pad(mer_s1, ((0, 0), (1, 1)), mode='edge')
        s1_mer_data['surfacecolor'] = mer_s1[mer_theta_pick,:]
        print('past meridian')

        #Get shell slice data
        s1_S_r095R = dsets[shell_field][ni]
        shell_s1 = s1_S_r095R.squeeze()
        shell_s1 -= np.mean(shell_s1)
        shell_s1 /= np.std(shell_s1)
        s1_shell1_data['surfacecolor'] = shell_s1[shell1_phi_pick_field,:][:,shell1_theta_pick_field]
        s1_shell2_data['surfacecolor'] = shell_s1[shell2_phi_pick_field,:][:,shell2_theta_pick_field]
        print('past shell')

        if first: #static colorbar
            minmax_s1 = np.array((2*np.std(eq_field_s1),))
            plotter.comm.Allreduce(MPI.IN_PLACE, minmax_s1, op=MPI.MAX)
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            norm = matplotlib.colors.Normalize(vmin=-minmax_s1[0], vmax=minmax_s1[0])

        data_dicts = [s1_shell1_data, s1_shell2_data, s1_mer_data, s1_eq_data]
#        data_dicts = [s1_mer_data, s1_eq_data, s1_shell2_data, s1_shell1_data]
#        data_dicts = [s1_eq_data]
        
        for i, d in enumerate(data_dicts):
            print('plotting data {}'.format(i))
            x = d['x']
            y = d['y']
            z = d['z']
            sfc = cmap(norm(d['surfacecolor']))
            surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
            d['surf'] = surf

            if first:
                cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', format=FormatStrFormatter('%.1f'))
                cbar.set_label('normalized s1' + ', t = {:.3e}'.format(time_data['sim_time'][ni]))
            ax.view_init(azim=125, elev=25)
            ax.set_xlim(-0.7*r_outer, 0.7*r_outer)
            ax.set_ylim(-0.7*r_outer, 0.7*r_outer)
            ax.set_zlim(-0.7*r_outer, 0.7*r_outer)
            ax.axis('off')
        fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, time_data['write_number'][ni]), dpi=200, bbox_inches='tight')

        gc.collect()
        ax.clear()
        first = False
