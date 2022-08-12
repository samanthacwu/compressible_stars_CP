

import os
import shelve
import pstats
import numpy as np


# Module constants
joined_filename = 'joined_stats.db'
summed_filename = 'summed_stats.prof'


def set_plot_defaults(rcParams):

    import matplotlib.pyplot as plt
    bar_colors = plt.get_cmap('Paired')

    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    #rcParams['axes.color_cycle'] = bar_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.facecolor'] = 'white'
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'white'
    #rcParams['patch.facecolor'] = bar_colors[0]
#    rcParams['font.family'] = 'StixGeneral'

def make_graph(profile, output_file, node_thresh=0.5):

    import subprocess

    proc_graph = subprocess.Popen(["./gprof2dot", "--skew", "0.5", "-n", "{:f}".format(node_thresh),
                                   "-f", "pstats", profile],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env={"LANG" : "en_US.UTF-8"})


    # the directed graph is produced by proc_graph.stdout
    proc_dot = subprocess.Popen(["dot", "-Tpng", "-o", output_file+'.png'],
                                stdin = proc_graph.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env={"LANG" : "en_US.UTF-8"})

    stdout, stderr = proc_dot.communicate()
    print(stdout, stderr)
    proc_dot = subprocess.Popen(["dot", "-Tpdf", "-o", output_file+'.pdf'],
                                stdin = proc_graph.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env={"LANG" : "en_US.UTF-8"})

    stdout, stderr = proc_dot.communicate()
    print(stdout, stderr)


def sort_dict(dict_to_sort):
    sorted_list = sorted(dict_to_sort.items(), key=lambda data_i: test_criteria(data_i[1]), reverse=True)
    return sorted_list

def sort_by_total(joined_stat):
    return sorted(joined_stat.items(), key=lambda kv: np.sum(kv[1]), reverse=True)

def test_criteria(data):
    return np.max(data)

def clean_display(ax):
    # from http://nbviewer.ipython.org/gist/anonymous/5357268
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')


def make_pdf(stats_pdf_dict, total_time, label='', N_profiles=50, thresh=0.005, verbose=False):

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    set_plot_defaults(rcParams)

    sorted_list = sort_by_total(stats_pdf_dict)

    composite_data_set = []
    composite_label = []
    composite_key_label = []

    fig_stacked = plt.figure()
    ax_stacked = fig_stacked.add_subplot(1,1,1)

    i_mpi_list = []
    i_fftw_list = []
    i_fft_list = []
    fft_type_list = ["ifft", "_dct", "rfft"]
    exclude_list = ["load_dynamic", "__init__", "<frozen", "importlib"]

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break

        if "gssv" in func[2]:
            if verbose:
                print("found sparse solve call:",func[2], " at ", i_sort)
            i_gssv = i_sort

        if "mpi4py.MPI" in func[2]:
            if verbose:
                print("found MPI call:",func[2], " at ", i_sort)
            i_mpi_list.append(i_sort)

    # bubble sparse solve to the top
    if "gssv" in func[2]:
        sorted_list.insert(0,sorted_list.pop(i_gssv))
    last_insert = 0
    # insert MPI calls next
    for i_resort in i_mpi_list:
            sorted_list.insert(last_insert+1,sorted_list.pop(i_resort))
            if verbose:
                print("moved entry {:d}->{:d}".format(i_resort, last_insert+1))
            last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if "fftw.fftw_wrappers.Transpose" in func[2]:
            if verbose:
                print("found fftw transpose call:",func[2], " at ", i_sort)
            sorted_list.insert(last_insert+1,sorted_list.pop(i_sort))
            if verbose:
                print("moved entry {:d}->{:d}".format(i_sort, last_insert+1))
            last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if any(fft_type in func[2] for fft_type in fft_type_list):
            if verbose:
                print("found fft call:",func[2], " at ", i_sort)
            if i_sort < N_profiles:
                sorted_list.insert(last_insert+1,sorted_list.pop(i_sort))
                if verbose:
                    print("moved entry {:d}->{:d}".format(i_sort, last_insert+1))
                last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break
        if any((exclude_type in func[0] or exclude_type in func[2]) for exclude_type in exclude_list):
            if verbose:
                print("found excluded call:",func[2], " at ", i_sort, " ... popping.")
            sorted_list.pop(i_sort)


    routine_text = "top {:d} routines for {:s}".format(N_profiles, label)
    if verbose:
        print()
        print("{:80s}".format(routine_text),"     min      mean       max   (mean%total)   (m%t cum.)")
        print(120*"-")
    running=0

    for i_fig, (func, data_list) in enumerate(sorted_list):
        data = np.array(data_list)
        N_data = data.shape[0]

        # if i_fig == N_profiles:
        #     break
        # if test_criteria(data)/total_time < thresh:
        #     break
        if i_fig+1 == N_profiles or (i_fig > last_insert and test_criteria(data)/total_time < thresh):
            break

        if i_fig == 0:
            previous_data = np.zeros_like(data)

        N_missing = previous_data.size - data.size

        if N_missing != 0:
            if verbose:
                print("missing {:d} values; setting to zero".format(N_missing))
            for i in range(N_missing):
                data_list.insert(N_missing*(i+1)-1, 0)
            data = np.array(data_list)
            N_data = data.shape[0]

        if func[0] == '~':
            title_string = func[2]
        else:
            title_string = "{:s}:{:d}:{:s}".format(*func)

        def percent_time(sub_time):
            sub_string = "{:4.2f}%".format(100*sub_time/total_time)
            return sub_string

        running += np.mean(data)
        timing_data_string = "{:8.2g} |{:8.2g} |{:8.2g}  ({:s}) ({:s})".format(np.min(data), np.mean(data), np.max(data), percent_time(np.mean(data)), percent_time(running))

        if verbose:
            print("{:80s} = {:s}".format(title_string, timing_data_string))

        timing_data_string = "min {:s} | {:s} | {:s} max".format(percent_time(np.min(data)), percent_time(np.mean(data)), percent_time(np.max(data)))

        title_string += "\n{:s}".format(timing_data_string)

        key_label = "{:s} {:s}".format(percent_time(np.mean(data)),func[2])
        short_label = "{:s}".format(percent_time(np.mean(data)))

        composite_data_set.append([data])
        composite_label.append(short_label)
        composite_key_label.append(key_label)


        if N_data > 200:
            N_bins = 100
            logscale = True
        else:
            N_bins = int(np.ceil(N_data/4))
            logscale = False

        q_color = next(ax_stacked._get_lines.prop_cycler)['color']

        fig = plt.figure()

        # pdf plot over many cores
        ax1 = fig.add_subplot(1,2,1)

        #hist_values, bin_edges = np.histogram(data, bins=N_bins)
        #ax1.barh(hist_values, bin_edges[1:])
        ax1.hist(data, bins=N_bins, orientation='horizontal', log=logscale, linewidth=0, color=q_color)
        ax1.set_xlabel("N cores/bin")
        ax1.set_ylabel("time (sec)")
        ax1.grid(axis = 'x', color ='white', linestyle='-')


        # bar plot for each core
        ax2 = fig.add_subplot(1,2,2)
        ax2.bar(np.arange(N_data), data, linewidth=0, width=1, color=q_color)
        ax2.set_xlim(-0.5, N_data+0.5)
        ax2.set_xlabel("core #")
        clean_display(ax2)

        ax2.grid(axis = 'y', color ='white', linestyle='-')

        # end include

        ax1.set_ylim(0, 1.1*np.max(data))
        ax2.set_ylim(0, 1.1*np.max(data))


        fig.suptitle(title_string)
        fig.savefig(label+'_{:06d}.png'.format(i_fig+1), dpi=200)
        plt.close(fig)

        ax_stacked.bar(np.arange(N_data), data, bottom=previous_data, label=short_label, linewidth=0,
                       width=1, color=q_color)
        previous_data += data

    clean_display(ax_stacked)
    ax_stacked.set_xlim(-0.5, N_data+0.5)
    ax_stacked.set_xlabel('core #')
    ax_stacked.set_ylabel('total time (sec)')
    ax_stacked.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    ax_stacked.set_title("per core timings for routines above {:g}% total time".format(thresh*100))
    ax_stacked.grid(axis = 'y', color ='white', linestyle='-')
    points_per_data = 10
    fig_x_size = 10
    fig_stacked.savefig(label+"_per_core_timings.png", dpi=max(200, N_data*points_per_data/fig_x_size))
    plt.close(fig_stacked)


    # pdf plot over many cores
    fig_composite = plt.figure()
    ax_composite = fig_composite.add_subplot(1,1,1)

    composite_data_set = np.squeeze(composite_data_set)
    print(composite_data_set.shape)
    n, bins, patches = ax_composite.hist(composite_data_set, bins=N_bins, orientation='vertical', log=logscale, linewidth=0, stacked=True,
                                         label=composite_label)

    clean_display(ax_composite)
    ax_composite.grid(axis = 'y', color ='white', linestyle='-')

    ax_composite.set_ylabel("N cores/bin")
    ax_composite.set_xlabel("total time (sec)")
    ax_composite.set_ylim(0, 1.1*np.max(composite_data_set))
    ax_composite.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)

    fig_composite.suptitle("composite PDF for routines above {:g}% total time".format(thresh*100))
    fig_composite.savefig(label+'_composite.png', dpi=200)
    plt.close(fig_composite)

    fig_key = plt.figure()
    plt.figlegend(patches, composite_key_label, 'center')
    #ax_key.legend(loc='center')
    fig_key.savefig(label+"_composite_key.png")
    plt.close(fig_key)


def combine_profiles(directory, filenames, verbose=False):
    """Combine statistics from a collection of profiles."""

    from collections import defaultdict
    from contextlib import closing

    summed_stats = pstats.Stats()
    farray = lambda: np.zeros(len(filenames), dtype=np.float64)
    joined_primcalls = defaultdict(farray)
    joined_totcalls = defaultdict(farray)
    joined_tottime = defaultdict(farray)
    joined_cumtime = defaultdict(farray)

    if verbose:
        print("Combining profiles:")

    for i, filename in enumerate(filenames):
        if verbose:
            print("  {:s}".format(filename))

        stats = pstats.Stats(filename)
        stats.strip_dirs()
        summed_stats.add(stats)

        for func, (primcalls, totcalls, tottime, cumtime, callers) in stats.stats.items():
            joined_primcalls[func][i] = primcalls
            joined_totcalls[func][i] = totcalls
            joined_tottime[func][i] = tottime
            joined_cumtime[func][i] = cumtime

    n_processes = len(filenames)
    average_runtime = summed_stats.total_tt / n_processes
    if verbose:
        print("  Average runtime: {:g} s".format(average_runtime))
    summed_stats.dump_stats(os.path.join(directory, summed_filename))

    with closing(shelve.open(os.path.join(directory, joined_filename), flag='n')) as shelf:
        shelf['primcalls'] = dict(joined_primcalls)
        shelf['totcalls'] = dict(joined_totcalls)
        shelf['tottime'] = dict(joined_tottime)
        shelf['cumtime'] = dict(joined_cumtime)
        shelf['average_runtime'] = average_runtime
        shelf['n_processes'] = n_processes


def read_database(directory):

    from contextlib import closing

    summed_stats = pstats.Stats(os.path.join(directory, summed_filename))

    with closing(shelve.open(os.path.join(directory, joined_filename), flag='r')) as shelf:
        primcalls = shelf['primcalls']
        totcalls = shelf['totcalls']
        tottime = shelf['tottime']
        cumtime = shelf['cumtime']
        average_runtime = shelf['average_runtime']
        n_processes = shelf['n_processes']

    return summed_stats, primcalls, totcalls, tottime, cumtime, average_runtime, n_processes


def natural_sort(l):

    import re

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


if __name__ == "__main__":

    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Analyze parallel python profiles.")
    parser.add_argument('command', choices=['process', 'plot'], help="Combine profiles into database, or plot database")
    parser.add_argument('directory', nargs='?', default='.', help="Directory containing profiles / database")
    parser.add_argument('--pattern', default='prof.*', help="Profile naming pattern (e.g. prof.*)")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    if args.command == 'process':
        pathname = os.path.join(args.directory, args.pattern)
        filenames = natural_sort(glob.glob(pathname))
        if filenames:
            combine_profiles('.', filenames, verbose=args.verbose)
        else:
            raise ValueError("No profiles found.")
    elif args.command == 'plot':
        # PDFs
        #summed_stats, primcalls, totcalls, tottime, cumtime, average_runtime, n_processes = read_database(args.directory)
        #make_pdf(tottime, average_runtime, label="tt", verbose=args.verbose)
        # Graphs
        summed_path = os.path.join(args.directory, summed_filename)
        make_graph(summed_path, 'full_code_profile', node_thresh=0.1)
        make_graph(summed_path, 'above_1_percent', node_thresh=1)
        make_graph(summed_path, 'above_5_percent', node_thresh=5)
    else:
        raise ValueError("Error parsing commands.")
