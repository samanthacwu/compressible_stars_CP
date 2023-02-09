import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap

from plotpal.slices import SlicePlotter
from plotpal.file_reader import match_basis

# Define smooth Heaviside functions
from scipy.special import erf 
def one_to_zero(x, x0, width=0.1):
        return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
        return -(one_to_zero(*args, **kwargs) - 1)

r_outer = 2
sponge_function = lambda r: zero_to_one(r, r_outer - 0.15, 0.07)



turb_root_dir = 'twoRcore_re3e4_damping/'
turb_start_file = 10
wave_start_file = 150
wave_root_dir = 'compressible_re1e4_waves/'
file_dir='slices'
out_name='dynamics_figure'
n_files = 10
start_fig = 1

turb_plotter = SlicePlotter(turb_root_dir, file_dir=file_dir, out_name=out_name, start_file=turb_start_file, n_files=n_files)
wave_plotter = SlicePlotter(wave_root_dir, file_dir=file_dir, out_name=out_name, start_file=wave_start_file, n_files=n_files)

turb_tasks_cz = ['equator(u_B)',]
turb_tasks = ['equator(u_B)', 'equator(u_S1)']
wave_tasks = ['equator(u_B)', 'equator(u_S1)', 'equator(u_S2)']
turb_coords_cz = dict()
turb_coords = dict()
wave_coords = dict()
for d in [turb_coords, turb_coords_cz, wave_coords]:
    for key in ['r', 'phi', 'rr', 'pp']:
        d[key] = []

true_maxmin = 8e-3

first = True
with turb_plotter.my_sync:
    while turb_plotter.writes_remain() and wave_plotter.writes_remain():
        turb_dsets, turb_ni = turb_plotter.get_dsets(turb_tasks)
        wave_dsets, wave_ni = wave_plotter.get_dsets(wave_tasks)
        if first:
            for d, dsets, tasks in zip((turb_coords, turb_coords_cz, wave_coords), (turb_dsets, turb_dsets, wave_dsets), \
                                        (turb_tasks, turb_tasks_cz, wave_tasks)):
                for i, task in enumerate(tasks):
                    d['r'].append(match_basis(dsets[task], 'r'))
                    d['phi'].append(match_basis(dsets[task], 'phi'))
                    d['phi'][-1] = np.append(d['phi'][-1], np.array([d['phi'][-1][0] + 2*np.pi]))
                    rr, pp = np.meshgrid(d['r'][-1], d['phi'][-1])
                    d['rr'].append(rr)
                    d['pp'].append(pp)
                full_rr = np.concatenate(d['rr'], axis=1)
                full_pp = np.concatenate(d['pp'], axis=1)
                d['xx'] = full_rr*np.cos(full_pp)
                d['yy'] = full_rr*np.sin(full_pp)

        fig = plt.figure(figsize=(7.5, 3))
        ax1 = fig.add_axes([0, 0, 0.3, 0.9], polar=False)
        ax2 = fig.add_axes([0.35, 0, 0.3, 0.9], polar=False)
        ax3 = fig.add_axes([0.7, 0, 0.3, 0.9], polar=False)
        cax1 = fig.add_axes([0.05, 0.97, 0.2, 0.03])
        cax2 = fig.add_axes([0.575, 0.97, 0.2, 0.03])
        plots = []



        for ax, d, dsets, ni, tasks, i in zip((ax1, ax2, ax3), (turb_coords_cz, turb_coords, wave_coords), \
                                    (turb_dsets, turb_dsets, wave_dsets), (turb_ni, turb_ni, wave_ni), \
                                    (turb_tasks_cz, turb_tasks, wave_tasks), (0, 1, 2)):
            data = []
            for t in tasks:
                data.append(dsets[t][ni,2,:].squeeze())
            data = np.concatenate(data, axis=1)
            if i == 0:
                vmin = -true_maxmin
                vmax = true_maxmin
            else:
                data -= np.mean(data, axis=0)[None,:]
                data /= np.std(data, axis=0)
                vmin = -2
                vmax = 2

            plot = ax.pcolormesh(d['xx'], d['yy'], data, cmap='PuOr_r', shading='nearest', rasterized=True, vmin=vmin, vmax=vmax)
            plots.append(plot)
            if i == 1:
                #add grey mask over damping region
                pmask = sponge_function(np.sqrt(d['xx']**2 +  d['yy']**2))
                t_cmap = np.ones([256, 4])*0.7
                t_cmap[:, 3] = np.linspace(0, 0.2, 256)
                t_cmap = ListedColormap(t_cmap)
                color2 = ax.pcolormesh(d['xx'], d['yy'], pmask, shading='auto', cmap=t_cmap, vmin=0, vmax=1, rasterized=True)
            if i == 0:
                cbar = plt.colorbar(plot, cax=cax1, orientation='horizontal')
                cax1.text(-0.05, 0.5, r'$u_r$', transform=cax1.transAxes, va='center', ha='right')
            elif i == 1:
                cbar = plt.colorbar(plot, cax=cax2, orientation='horizontal')
                cax2.text(-0.05, 0.5, r'$u_r/\sigma(u_r)$', transform=cax2.transAxes, va='center', ha='right')
            if i in [0, 1]:
                cbar.set_ticks((vmin, 0, vmax))
                cbar.set_ticklabels(['{:.2f}'.format(vmin), '0', '{:.2f}'.format(vmax)])
            ax.set_yticks([])
            ax.set_xticks([])
            for direction in ['left', 'right', 'bottom', 'top']:
                ax.spines[direction].set_visible(False)
            outline_r = d['r'][-1].max()
            outline_phi = np.linspace(0, 2.1*np.pi, 1000)
            ax.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5)
            if i == 0:
                ax2.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5)
                phi_1 = np.pi*0.45
                xy1_top = outline_r*np.array((np.cos(phi_1), np.sin(phi_1)))
                xy2_top = (0, outline_r)
                xy1_bot = outline_r*np.array((np.cos(-phi_1), np.sin(-phi_1)))
                xy2_bot = (0, -outline_r)
                for xy1, xy2 in zip((xy1_top, xy1_bot),(xy2_top, xy2_bot)):
                    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                                                  axesA=ax, axesB=ax2, color="black", lw=0.5)
                    ax2.add_artist(con)
            if i == 1:
                ax3.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5)
                phi_1 = np.pi*0.45
                xy1_top = outline_r*np.array((np.cos(phi_1), np.sin(phi_1)))
                xy2_top = (0, outline_r)
                xy1_bot = outline_r*np.array((np.cos(-phi_1), np.sin(-phi_1)))
                xy2_bot = (0, -outline_r)
                for xy1, xy2 in zip((xy1_top, xy1_bot),(xy2_top, xy2_bot)):
                    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                                                  axesA=ax, axesB=ax3, color="black", lw=0.5)
                    ax3.add_artist(con)


            ax.set_xlim(-outline_r*1.01, outline_r*1.01)
            ax.set_ylim(-outline_r*1.01, outline_r*1.01)

        write_num = turb_plotter.current_file_handle['scales/write_number'][turb_ni] 
        figname = '{:s}/{:s}_{:06d}.png'.format(turb_plotter.out_dir, turb_plotter.out_name, int(write_num+start_fig-1))
        fig.savefig(figname, dpi=300, bbox_inches='tight')

        first = False


