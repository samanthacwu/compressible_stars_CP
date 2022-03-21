"""
This script plots nice 3D, fancy plots of ball-shell stars.

Usage:
    ballShell_plot_volume_visual.py <root_dir> [options]

Options:
    --data_dir=<dir>     Name of data handler directory [default: slices]
    --r_outer=<r>        Value of r at outer boundary [default: 3.21]
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
import matplotlib.pyplot as plt

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


def spherical_to_cartesian(phi, theta, r):
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

r_outer = float(args['--r_outer'])

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
    fields = ['s1_B(phi=0)', 's1_B(phi=1.5*pi)', 's1_B(r=1)', 's1_eq_B', 
              's1_S1(phi=0)', 's1_S1(phi=1.5*pi)', 's1_S1(r=0.95R)', 's1_eq_S1',
              's1_S2(phi=0)', 's1_S2(phi=1.5*pi)', 's1_S2(r=0.95R)', 's1_eq_S2']
    bases  = ['r', 'phi', 'theta']

    re = float(root_dir.split('Re')[-1].split('_')[0])

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 0.9],projection='3d')
    cax = fig.add_axes([0.05, 0.91, 0.9, 0.08])

    s1_eq_data = OrderedDict()
    s1_mer_data = OrderedDict()
    s1_shell_data = OrderedDict()

    equator_line = OrderedDict()
    near_equator_line = OrderedDict()
    mer_line = OrderedDict()
    near_mer_line = OrderedDict()
    lines = [equator_line, near_equator_line, mer_line, near_mer_line]

    colorbar_dict=dict(lenmode='fraction', len=0.75, thickness=20)
    r_arrays = []

    phis = 0
    phie = 3*np.pi/2

    bs = {}
    first = True
    while plotter.writes_remain():
        dsets, ni = plotter.get_dsets(fields)
        time_data=dsets[fields[0]].dims[0]
        theta = match_basis(dsets['s1_B(r=1)'], 'theta')
        phi   = match_basis(dsets['s1_B(r=1)'], 'phi')
        phi_de   = match_basis(dsets['s1_eq_B'], 'phi')
        theta_de = match_basis(dsets['s1_B(phi=0)'], 'theta')
        rB_de    = match_basis(dsets['s1_B(phi=0)'], 'r')
        rS1_de   = match_basis(dsets['s1_S1(phi=0)'], 'r')
        rS2_de   = match_basis(dsets['s1_S2(phi=0)'], 'r')
        r_de = np.concatenate((rB_de, rS1_de, rS2_de), axis=-1)
        dphi = phi[1] - phi[0]
        dphi_de = phi_de[1] - phi_de[0]

        phi_vert_de, theta_vert_de, r_vert_de = build_spherical_vertices(phi_de, theta_de, r_de, 0, r_outer)
        phi_vert, theta_vert, r_vert_de = build_spherical_vertices(phi, theta, r_de, 0, r_outer)
        phi_vert_pick = (phis < phi_vert)*(phi_vert < phie)
        phi_vert_pick_de = (phis < phi_vert_de)*(phi_vert_de < phie)
        phi_vert = np.concatenate([[phis], phi_vert[phi_vert_pick], [phie]], axis=0)
        phi_vert_de = np.concatenate([[phis], phi_vert_de[phi_vert_pick_de], [phie]], axis=0)
        xo, yo, zo = spherical_to_cartesian(phi_vert, theta_vert, [0.95*r_outer])[:,:,:,0]
        xs, ys, zs = spherical_to_cartesian([phis], theta_vert_de, r_vert_de)[:,0,:,:]
        xe, ye, ze = spherical_to_cartesian([phie], theta_vert_de, r_vert_de)[:,0,:,:]
#        theta  = np.pad(plain_theta, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi, 0))
#        phi    = np.pad(plain_phi, ((0,1), (0,0), (0,0)), mode='constant', constant_values=(2*np.pi))
#        theta_de  = np.pad(plain_theta_de, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi, 0))
#        phi_de    = np.pad(plain_phi_de, ((0,1), (0,0), (0,0)), mode='constant', constant_values=(2*np.pi))
#        rB_de     = np.pad(plain_rB_de, ((0,0), (0,0), (1,0)), mode='constant', constant_values=(0,))
#        rS_de     = np.pad(plain_rS_de, ((0,0), (0,0), (0,1)), mode='constant', constant_values=(r_outer,))

        mean_s1_B  = np.expand_dims(np.mean(dsets['s1_eq_B'][ni], axis=0), axis=0)
        mean_s1_S1 = np.expand_dims(np.mean(dsets['s1_eq_S1'][ni], axis=0), axis=0)
        mean_s1_S2 = np.expand_dims(np.mean(dsets['s1_eq_S2'][ni], axis=0), axis=0)
        s1_eq_B = dsets['s1_eq_B'][ni] - mean_s1_B
        s1_eq_S1 = dsets['s1_eq_S1'][ni] - mean_s1_S1
        s1_eq_S2 = dsets['s1_eq_S2'][ni] - mean_s1_S2
        s1_B_r1 = dsets['s1_B(r=1)'][ni] - np.expand_dims(np.mean(np.mean(dsets['s1_B(r=1)'][ni], axis=2), axis=1), axis=[1,2])
#        s1_S_r095R = dsets['s1_S1(r=0.95R)'][ni] - np.expand_dims(np.mean(np.mean(dsets['s1_S1(r=0.95R)'][ni], axis=2), axis=1), axis=[1,2])
        s1_S_r095R = dsets['s1_S2(r=0.95R)'][ni] - np.expand_dims(np.mean(np.mean(dsets['s1_S2(r=0.95R)'][ni], axis=2), axis=1), axis=[1,2])
        phi_pick = (phis-dphi_de/2 < phi)*(phi < phie + dphi_de/2)
        s1_S_r095R = s1_S_r095R[:,:,0][phi_pick]
        s1_B_phi0  = dsets['s1_B(phi=0)'][ni] - mean_s1_B 
        s1_B_phipi = dsets['s1_B(phi=1.5*pi)'][ni] - mean_s1_B 
        s1_S1_phi0  = dsets['s1_S1(phi=0)'][ni] - mean_s1_S1
        s1_S1_phipi = dsets['s1_S1(phi=1.5*pi)'][ni] - mean_s1_S1
        s1_S2_phi0  = dsets['s1_S2(phi=0)'][ni] - mean_s1_S2
        s1_S2_phipi = dsets['s1_S2(phi=1.5*pi)'][ni] - mean_s1_S2
#        for eq_data in [s1_eq_data]:
#            eq_data['x'] = (full_r_de*np.cos(phi_de)).squeeze()
#            eq_data['y'] = (full_r_de*np.sin(phi_de)).squeeze()
#            eq_data['z'] = np.zeros_like(eq_data['x'])

#        X_mer1_de = (full_r_de*np.cos(0)*np.sin(theta_de)).squeeze()
#        Y_mer1_de = np.zeros_like(X_mer1_de)
#        Z_mer1_de = (full_r_de*np.cos(theta_de)).squeeze()
#        X_mer2_de = (full_r_de*np.cos(np.pi)*np.sin(theta_de)).squeeze()
#        Y_mer2_de = np.zeros_like(X_mer2_de)
#        Z_mer2_de = (full_r_de*np.cos(theta_de)).squeeze()
#
#
#        for mer_data in [s1_mer_data]:
#            mer_data['x'] = np.concatenate((X_mer1_de, X_mer2_de))
#            mer_data['y'] = np.concatenate((Y_mer1_de, Y_mer2_de))
#            mer_data['z'] = np.concatenate((Z_mer1_de, Z_mer2_de))
#
#        for shell_data in [s1_shell_data,]:
#            shell_data['x'] = 0.95*r_outer*(np.cos(plain_phi)*np.sin(plain_theta)).squeeze()
#            shell_data['y'] = 0.95*r_outer*(np.sin(plain_phi)*np.sin(plain_theta)).squeeze()
#            shell_data['z'] = 0.95*r_outer*(np.ones_like(plain_phi)*np.cos(plain_theta)).squeeze()
#
#
#        equator_line['x'] = r_outer*(np.cos(phi)).flatten()
#        equator_line['y'] = r_outer*(np.sin(phi)).flatten()
#        equator_line['z'] = np.zeros_like(equator_line['x'])
#        for k in ['x', 'z', 'y']:
#            near_equator_line[k] = 0.955*equator_line[k]
#            near_equator_line[k] = near_equator_line[k][equator_line['y'] < 0]
#
#        mer_angle = np.linspace(0, 2.1*np.pi, 100)
#        mer_line['x'] = r_outer*np.sin(mer_angle)
#        mer_line['y'] = r_outer*np.zeros_like(mer_line['x'])
#        mer_line['z'] = r_outer*np.cos(mer_angle)
#        for k in ['x', 'y', 'z']:
#            near_mer_line[k] = 0.955*mer_line[k]
#            near_mer_line[k] = near_mer_line[k][mer_line['z'] < 0]
#
#
#        shell_bool = np.logical_or(shell_data['y'] < 0, shell_data['z'] < 0)
#        for k in ['x', 'y', 'z']:
#            for shell_data in [s1_shell_data,]:
#                shell_data[k] = np.where(shell_bool, shell_data[k], 0)
#

#        eq_field_s1_B = np.pad(s1_eq_B.squeeze(), ((0,1), (1,0)), mode='edge')
#        eq_field_s1_S = np.pad(s1_eq_S1.squeeze(), ((0,1), (0,1)), mode='edge')
        eq_field_s1 = np.concatenate((s1_eq_B, s1_eq_S1, s1_eq_S2), axis=-1)
        radial_scaling = np.sqrt(np.mean(eq_field_s1**2, axis=0))
        eq_field_s1 /= radial_scaling
        minmax_s1 = 2*np.std(eq_field_s1)
        s1_eq_data['surfacecolor'] = eq_field_s1
#        mer_0_s1_B = np.pad(s1_B_phi0.squeeze(), ((1, 1), (1, 0)), mode='edge') 
#        mer_1_s1_B = np.pad(s1_B_phipi.squeeze(), ((1, 1), (1, 0)), mode='edge') 
#        mer_0_s1_S = np.pad(s1_S_phi0.squeeze(), ((1, 1), (1, 0)), mode='edge') 
#        mer_1_s1_S = np.pad(s1_S_phipi.squeeze(), ((1, 1), (1, 0)), mode='edge') 
        mer_0_s1_B = s1_B_phi0.squeeze() 
        mer_1_s1_B = s1_B_phipi.squeeze()
        mer_0_s1_S1 = s1_S1_phi0.squeeze()
        mer_1_s1_S1 = s1_S1_phipi.squeeze()
        mer_0_s1_S2 = s1_S2_phi0.squeeze()
        mer_1_s1_S2 = s1_S2_phipi.squeeze()

        mer_0_s1 = np.concatenate((mer_0_s1_B, mer_0_s1_S1, mer_0_s1_S2), axis=-1)/radial_scaling
        mer_1_s1 = np.concatenate((mer_1_s1_B, mer_1_s1_S1, mer_1_s1_S2), axis=-1)/radial_scaling
        mer_s1 = np.concatenate([mer_1_s1.transpose((1,0))[::-1], mer_0_s1.transpose((1,0))], axis=0)
#        mer_s1   = np.concatenate((mer_0_s1, mer_1_s1))

        x_mer = np.concatenate([xe.T[::-1], xs.T[1:]], axis=0)
        y_mer = np.concatenate([ye.T[::-1], ys.T[1:]], axis=0)
        z_mer = np.concatenate([ze.T[::-1], zs.T[1:]], axis=0)
        s1_mer_data['x'] = x_mer
        s1_mer_data['y'] = y_mer
        s1_mer_data['z'] = z_mer
        
        s1_shell_data['x'] = xo
        s1_shell_data['y'] = yo
        s1_shell_data['z'] = zo
        

        s1_mer_data['surfacecolor'] = mer_s1
        shell_s1 = s1_S_r095R.squeeze()
        shell_s1 /= np.sqrt(np.mean(shell_s1**2))
        s1_shell_data['surfacecolor'] = shell_s1#np.where(shell_bool, shell_s1, 0)

        cmap = matplotlib.cm.get_cmap('RdBu_r')
        norm = matplotlib.colors.Normalize(vmin=-minmax_s1, vmax=minmax_s1)
        if first:
            for i, d in enumerate([s1_mer_data, s1_shell_data]):
                #I think it looks nicer with z and x swapped
                x = d['z']
                y = d['y']
                z = d['x']
                sfc = cmap(norm(d['surfacecolor']))
                print(x.shape, y.shape, z.shape, sfc.shape)
                surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
                d['surf'] = surf
    #            ax.plot_wireframe(x, y, z, ccount=1, rcount=1, linewidth=1, color='black')
            ax.set_xlim(-0.7*r_outer, 0.7*r_outer)
            ax.set_ylim(-0.7*r_outer, 0.7*r_outer)
            ax.set_zlim(-0.7*r_outer, 0.7*r_outer)
            ax.view_init(azim=-150, elev=20)
            ax.axis('off')
            first = False
        else:
            for i, d in enumerate([s1_mer_data, s1_shell_data]):
                sfc = cmap(norm(d['surfacecolor']))
                d['surf'].set_facecolors(sfc.reshape(sfc.size//4,4))
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        cbar.set_label('normalized s1' + ', t = {:.3e}'.format(time_data['sim_time'][ni]))

        fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, time_data['write_number'][ni]), dpi=200)
        for pax in [cax,]:
           pax.clear()
