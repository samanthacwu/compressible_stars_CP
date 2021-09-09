"""
This script plots nice 3D, fancy plots of ball-shell stars.

Usage:
    plot_ballShell_plotly3d.py <root_dir> [options]

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

from plotpal.file_reader import SingleFiletypePlotter as SFP

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

plotter = SFP(root_dir, file_dir=data_dir, fig_name='plotly3Dplot', start_file=1, n_files=np.inf, distribution='even')
if not plotter.idle:
    phi_keys = []
    theta_keys = []
    with h5py.File(plotter.files[0], 'r') as f:
        scale_keys = f['scales'].keys()
        for k in scale_keys:
            if k[0] == 'φ':
                phi_keys.append(k)
            if k[0] == 'θ':
                theta_keys.append(k)
    r_keys = ['r_0', 'r_1']

    fields = ['s1B(phi=0)', 's1B(phi=pi)', 's1B(r=1)', 's1B_eq', 's1S(phi=0)', 's1S(phi=pi)', 's1S(r=0.95R)', 's1S_eq']
    bases  = [r_keys[0], r_keys[1], phi_keys[0], theta_keys[0]]

    re = float(root_dir.split('Re')[-1].split('_')[0])

    fig = go.Figure(layout={'width': 2000, 'height': 1000})
    make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]], subplot_titles=['entropy fluctuations'], figure=fig, horizontal_spacing=0)
    fig.update_annotations(y=0.8, selector={'text':'T'})
    fig.update_layout(scene = {
                        'xaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'yaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'zaxis': {'showbackground':False, 'tickvals':[], 'title':''}},
#                      scene2 = {
#                        'xaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
#                        'yaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
#                        'zaxis': {'showbackground':False, 'tickvals':[], 'title':''}},
                      margin={'l':0, 'r':0, 'b':0, 't':50, 'pad':0}, font={'size' : 18}, annotations={'font' : {'size' : 16}})

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

    while plotter.files_remain(bases, fields):
        bs, tsk, write, time = plotter.read_next_file()
        plain_theta = bs[theta_keys[0]]
        plain_phi   = bs[phi_keys[0]]
        plain_rB     = bs[r_keys[0]]
        plain_rS     = bs[r_keys[1]]
        theta = np.pad(plain_theta, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi, 0))
        phi   = np.pad(plain_phi, ((0,1), (0,0), (0,0)), mode='constant', constant_values=(2*np.pi))
        rB     = np.pad(plain_rB, ((0,0), (0,0), (1,0)), mode='constant', constant_values=(0,))
        rS     = np.pad(plain_rS, ((0,0), (0,0), (0,1)), mode='constant', constant_values=(r_outer,))
        full_r = np.concatenate((rB, rS), axis=-1)

        mean_s1B = np.expand_dims(np.mean(tsk['s1B_eq'], axis=1), axis=1)
        mean_s1S = np.expand_dims(np.mean(tsk['s1S_eq'], axis=1), axis=1)
        tsk['s1B_eq'] -= mean_s1B
        tsk['s1S_eq'] -= mean_s1S
        tsk['s1B(r=1)'] -= np.expand_dims(np.mean(np.mean(tsk['s1B(r=1)'], axis=2), axis=1), axis=[1,2])
        tsk['s1S(r=0.95R)'] -= np.expand_dims(np.mean(np.mean(tsk['s1S(r=0.95R)'], axis=2), axis=1), axis=[1,2])
        tsk['s1B(phi=0)']  -= mean_s1B 
        tsk['s1B(phi=pi)'] -= mean_s1B 
        tsk['s1S(phi=0)']  -= mean_s1S
        tsk['s1S(phi=pi)'] -= mean_s1S
        for eq_data in [s1_eq_data]:
            eq_data['x'] = (full_r*np.cos(phi)).squeeze()
            eq_data['y'] = (full_r*np.sin(phi)).squeeze()
            eq_data['z'] = np.zeros_like(eq_data['x'])

        X_mer1 = (full_r*np.cos(0)*np.sin(theta)).squeeze()
        Y_mer1 = np.zeros_like(X_mer1)
        Z_mer1 = (full_r*np.cos(theta)).squeeze()
        X_mer2 = (full_r*np.cos(np.pi)*np.sin(theta)).squeeze()
        Y_mer2 = np.zeros_like(X_mer2)
        Z_mer2 = (full_r*np.cos(theta)).squeeze()

        for mer_data in [s1_mer_data]:
            mer_data['x'] = np.concatenate((X_mer1, X_mer2))
            mer_data['y'] = np.concatenate((Y_mer1, Y_mer2))
            mer_data['z'] = np.concatenate((Z_mer1, Z_mer2))

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


        for i, n in enumerate(write):
            print('plotting {}'.format(n))
            eq_field_s1B = np.pad(tsk['s1B_eq'][i,:].squeeze(), ((0,1), (1,0)), mode='edge')
            eq_field_s1S = np.pad(tsk['s1S_eq'][i,:].squeeze(), ((0,1), (0,1)), mode='edge')
            eq_field_s1 = np.concatenate((eq_field_s1B, eq_field_s1S), axis=-1)
            radial_scaling = np.sqrt(np.expand_dims(np.mean(eq_field_s1**2, axis=0), axis=0))
            eq_field_s1 /= radial_scaling
            minmax_s1 = 2*np.std(eq_field_s1)
            s1_eq_data['surfacecolor'] = eq_field_s1
            mer_0_s1B = np.pad(tsk['s1B(phi=0)'][i,:].squeeze(), ((1, 1), (1, 0)), mode='edge') 
            mer_1_s1B = np.pad(tsk['s1B(phi=pi)'][i,:].squeeze(), ((1, 1), (1, 0)), mode='edge') 
            mer_0_s1S = np.pad(tsk['s1S(phi=0)'][i,:].squeeze(), ((1, 1), (1, 0)), mode='edge') 
            mer_1_s1S = np.pad(tsk['s1S(phi=pi)'][i,:].squeeze(), ((1, 1), (1, 0)), mode='edge') 

            mer_0_s1 = np.concatenate((mer_0_s1B, mer_0_s1S), axis=-1)/radial_scaling
            mer_1_s1 = np.concatenate((mer_1_s1B, mer_1_s1S), axis=-1)/radial_scaling
            mer_s1   = np.concatenate((mer_0_s1, mer_1_s1))

            s1_mer_data['surfacecolor'] = mer_s1
            shell_s1 = tsk['s1S(r=0.95R)'][i,:].squeeze()
            shell_s1 /= np.sqrt(np.mean(shell_s1**2))
            s1_shell_data['surfacecolor'] = np.where(shell_bool, shell_s1, 0)

            fig.add_trace(go.Surface(**s1_shell_data, colorbar_x=-0.07, cmin=-minmax_s1, cmax=minmax_s1, colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)
            fig.add_trace(go.Surface(**s1_mer_data, colorbar_x=-0.07,   cmin=-minmax_s1, cmax=minmax_s1,   colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)
            fig.add_trace(go.Surface(**s1_eq_data, colorbar_x=-0.07,    cmin=-minmax_s1, cmax=minmax_s1,    colorscale='RdBu_r', colorbar=colorbar_dict), 1, 1)

            for l in lines:
                fig.add_trace(go.Scatter3d(**l, mode='lines', line={'color':'black'}, showlegend=False), 1, 1)

            title_text = 't = {:.3e}'.format(time[i])
            fig.update_layout(title=title_text)
            fig.update_annotations(y=0.85, selector={'text':title_text})
            figname = '{:s}/plotly3Dplot/plotly3Dplot_{:06d}.png'.format(root_dir, n)
            pio.write_image(fig, figname, format='png', engine='kaleido', scale=float(args['--scale']))
            fig.data = []
