"""
This script plots nice 3D, fancy plots of ball-shell stars.

Usage:
    ballShell_plot_volume_visual.py <root_dir> [options]

Options:
    --data_dir=<dir>     Name of data handler directory [default: slices]
    --r_outer=<r>        Value of r at outer boundary [default: 2.59]
    --scale=<s>          resolution scale factor [default: 1]
"""
from collections import OrderedDict

import h5py
import numpy as np
from docopt import docopt
args = docopt(__doc__)

from plotpal.file_reader import SingleTypeReader, match_basis

import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import plot_mpl
#import plotly.express as px

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

r_outer = float(args['--r_outer'])

fig_name='volume_vizualization'
plotter = SingleTypeReader(root_dir, data_dir, fig_name, start_file=1, n_files=np.inf, distribution='even-write')
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
    r_keys = ['r_0', 'r_1']

    fields = ['s1_B(phi=0)', 's1_B(phi=pi)', 's1_B(r=1)', 's1_B_eq', 's1_S(phi=0)', 's1_S(phi=pi)', 's1_S(r=0.95R)', 's1_S_eq']
    bases  = ['r', 'phi', 'theta']

    re = float(root_dir.split('Re')[-1].split('_')[0])

    fig = go.Figure(layout={'width': 1000, 'height': 1000})
    make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]], subplot_titles=['entropy fluctuations'], figure=fig, horizontal_spacing=0)
    fig.update_annotations(y=0.8, selector={'text':'T'})
    fig.update_layout(scene = {
                        'xaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'yaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'zaxis': {'showbackground':False, 'tickvals':[], 'title':''}},
                      margin={'l':0, 'r':0, 'b':0, 't':0, 'pad':0}, font={'size' : 18}, annotations={'font' : {'size' : 16}})

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

    bs = {}
    while plotter.writes_remain():
        dsets, ni = plotter.get_dsets(fields)
        time_data=dsets[fields[0]].dims[0]
        plain_theta = match_basis(dsets['s1_B(r=1)'], 'theta')[None,:,None]
        plain_phi   = match_basis(dsets['s1_B(r=1)'], 'phi')[:,None,None]
        plain_phi_de   = match_basis(dsets['s1_B_eq'], 'phi')[:,None,None]
        plain_theta_de = match_basis(dsets['s1_B(phi=0)'], 'theta')[None,:,None]
        plain_rB_de    = match_basis(dsets['s1_B(phi=0)'], 'r')[None,None,:]
        plain_rS_de    = match_basis(dsets['s1_S(phi=0)'], 'r')[None,None,:]
        theta  = np.pad(plain_theta, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi, 0))
        phi    = np.pad(plain_phi, ((0,1), (0,0), (0,0)), mode='constant', constant_values=(2*np.pi))
        theta_de  = np.pad(plain_theta_de, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi, 0))
        phi_de    = np.pad(plain_phi_de, ((0,1), (0,0), (0,0)), mode='constant', constant_values=(2*np.pi))
        rB_de     = np.pad(plain_rB_de, ((0,0), (0,0), (1,0)), mode='constant', constant_values=(0,))
        rS_de     = np.pad(plain_rS_de, ((0,0), (0,0), (0,1)), mode='constant', constant_values=(r_outer,))
        full_r_de = np.concatenate((rB_de, rS_de), axis=-1)

        mean_s1_B = np.expand_dims(np.mean(dsets['s1_B_eq'][ni], axis=0), axis=0)
        mean_s1_S = np.expand_dims(np.mean(dsets['s1_S_eq'][ni], axis=0), axis=0)
        s1_B_eq = dsets['s1_B_eq'][ni] - mean_s1_B
        s1_S_eq = dsets['s1_S_eq'][ni] - mean_s1_S
        s1_B_r1 = dsets['s1_B(r=1)'][ni] - np.expand_dims(np.mean(np.mean(dsets['s1_B(r=1)'][ni], axis=2), axis=1), axis=[1,2])
        s1_S_r095R = dsets['s1_S(r=0.95R)'][ni] - np.expand_dims(np.mean(np.mean(dsets['s1_S(r=0.95R)'][ni], axis=2), axis=1), axis=[1,2])
        s1_B_phi0  = dsets['s1_B(phi=0)'][ni] - mean_s1_B 
        s1_B_phipi = dsets['s1_B(phi=pi)'][ni] - mean_s1_B 
        s1_S_phi0  = dsets['s1_S(phi=0)'][ni] - mean_s1_S
        s1_S_phipi = dsets['s1_S(phi=pi)'][ni] - mean_s1_S
        for eq_data in [s1_eq_data]:
            eq_data['x'] = (full_r_de*np.cos(phi_de)).squeeze()
            eq_data['y'] = (full_r_de*np.sin(phi_de)).squeeze()
            eq_data['z'] = np.zeros_like(eq_data['x'])

        X_mer1_de = (full_r_de*np.cos(0)*np.sin(theta_de)).squeeze()
        Y_mer1_de = np.zeros_like(X_mer1_de)
        Z_mer1_de = (full_r_de*np.cos(theta_de)).squeeze()
        X_mer2_de = (full_r_de*np.cos(np.pi)*np.sin(theta_de)).squeeze()
        Y_mer2_de = np.zeros_like(X_mer2_de)
        Z_mer2_de = (full_r_de*np.cos(theta_de)).squeeze()


        for mer_data in [s1_mer_data]:
            mer_data['x'] = np.concatenate((X_mer1_de, X_mer2_de))
            mer_data['y'] = np.concatenate((Y_mer1_de, Y_mer2_de))
            mer_data['z'] = np.concatenate((Z_mer1_de, Z_mer2_de))

        for shell_data in [s1_shell_data,]:
            shell_data['x'] = 0.95*r_outer*(np.cos(plain_phi)*np.sin(plain_theta)).squeeze()
            shell_data['y'] = 0.95*r_outer*(np.sin(plain_phi)*np.sin(plain_theta)).squeeze()
            shell_data['z'] = 0.95*r_outer*(np.ones_like(plain_phi)*np.cos(plain_theta)).squeeze()


        equator_line['x'] = r_outer*(np.cos(phi)).flatten()
        equator_line['y'] = r_outer*(np.sin(phi)).flatten()
        equator_line['z'] = np.zeros_like(equator_line['x'])
        for k in ['x', 'z', 'y']:
            near_equator_line[k] = 0.955*equator_line[k]
            near_equator_line[k] = near_equator_line[k][equator_line['y'] < 0]

        mer_angle = np.linspace(0, 2.1*np.pi, 100)
        mer_line['x'] = r_outer*np.sin(mer_angle)
        mer_line['y'] = r_outer*np.zeros_like(mer_line['x'])
        mer_line['z'] = r_outer*np.cos(mer_angle)
        for k in ['x', 'y', 'z']:
            near_mer_line[k] = 0.955*mer_line[k]
            near_mer_line[k] = near_mer_line[k][mer_line['z'] < 0]


        shell_bool = np.logical_or(shell_data['y'] < 0, shell_data['z'] < 0)
        for k in ['x', 'y', 'z']:
            for shell_data in [s1_shell_data,]:
                shell_data[k] = np.where(shell_bool, shell_data[k], 0)


        eq_field_s1_B = np.pad(s1_B_eq.squeeze(), ((0,1), (1,0)), mode='edge')
        eq_field_s1_S = np.pad(s1_S_eq.squeeze(), ((0,1), (0,1)), mode='edge')
        eq_field_s1 = np.concatenate((eq_field_s1_B, eq_field_s1_S), axis=-1)
        radial_scaling = np.sqrt(np.expand_dims(np.mean(eq_field_s1**2, axis=0), axis=0))
        eq_field_s1 /= radial_scaling
        minmax_s1 = 2*np.std(eq_field_s1)
        s1_eq_data['surfacecolor'] = eq_field_s1
        mer_0_s1_B = np.pad(s1_B_phi0.squeeze(), ((1, 1), (1, 0)), mode='edge') 
        mer_1_s1_B = np.pad(s1_B_phipi.squeeze(), ((1, 1), (1, 0)), mode='edge') 
        mer_0_s1_S = np.pad(s1_S_phi0.squeeze(), ((1, 1), (1, 0)), mode='edge') 
        mer_1_s1_S = np.pad(s1_S_phipi.squeeze(), ((1, 1), (1, 0)), mode='edge') 

        mer_0_s1 = np.concatenate((mer_0_s1_B, mer_0_s1_S), axis=-1)/radial_scaling
        mer_1_s1 = np.concatenate((mer_1_s1_B, mer_1_s1_S), axis=-1)/radial_scaling
        mer_s1   = np.concatenate((mer_0_s1, mer_1_s1))

        s1_mer_data['surfacecolor'] = mer_s1
        shell_s1 = s1_S_r095R.squeeze()
        shell_s1 /= np.sqrt(np.mean(shell_s1**2))
        s1_shell_data['surfacecolor'] = np.where(shell_bool, shell_s1, 0)

        fig.add_trace(go.Surface(**s1_shell_data, colorbar_x=-0.07, cmin=-minmax_s1, cmax=minmax_s1, colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)
        fig.add_trace(go.Surface(**s1_mer_data, colorbar_x=-0.07,   cmin=-minmax_s1, cmax=minmax_s1,   colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)
        fig.add_trace(go.Surface(**s1_eq_data, colorbar_x=-0.07,    cmin=-minmax_s1, cmax=minmax_s1,    colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)

        for l in lines:
            fig.add_trace(go.Scatter3d(**l, mode='lines', line={'color':'black'}, showlegend=False), 1, 1)

        title_text = 't = {:.3e}'.format(time_data['sim_time'][ni])
        viewpoint = {'camera_eye' : {'x' : 0.7, 'y': 1.1, 'z' : 1}}
        fig.update_layout(title=title_text, scene=viewpoint)
        fig.update_annotations(y=0.85, selector={'text':title_text})
        this_fig_name = '{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, time_data['write_number'][ni])
        pio.write_image(fig, this_fig_name, format='png', engine='kaleido', scale=float(args['--scale']))
        fig.data = []
