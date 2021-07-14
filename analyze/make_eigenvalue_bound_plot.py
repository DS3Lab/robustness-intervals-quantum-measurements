import numpy as np
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns

from lib.robustness_interval import GramianEigenvalueInterval

font_size = 14
font_size_large = 18
sns.set_style('ticks')
colors = sns.color_palette()
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = font_size_large
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1.5

# colors
vqe_color = colors[0]
vqe_color_noiseless = colors[0]
vqe_color_noisy = colors[1]

interval_color = colors[1]

lw = 1.0


def lower_bounds_plots(results_dir_noisy, results_dir_noiseless, vqe_label, err_step, fid_step, xmin=1.0, save_as=None):
    """

    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe

    """
    try:
        stats_df_noiseless = pd.read_pickle(os.path.join(results_dir_noiseless, 'statistics.pkl'))
    except FileNotFoundError:
        stats_df_noiseless = None

    try:
        stats_df_noisy = pd.read_pickle(os.path.join(results_dir_noisy, 'statistics.pkl'))
    except FileNotFoundError:
        stats_df_noisy = None

    if stats_df_noisy is None and stats_df_noiseless is None:
        raise FileNotFoundError('no data found')

    # create empty df if None
    stats_df_noisy = pd.DataFrame(
        columns=stats_df_noiseless.columns) if stats_df_noisy is None else stats_df_noisy
    stats_df_noiseless = pd.DataFrame(
        columns=stats_df_noisy.columns) if stats_df_noiseless is None else stats_df_noiseless

    plot_df_noiseless = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'fidelity', 'lower_bounds', 'upper_bounds'])

    plot_df_noisy = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'fidelity', 'lower_bounds', 'upper_bounds'])

    # compute gramian eigenvalue intervals without noise
    for r in stats_df_noiseless.index:
        vqe_energy = np.mean(stats_df_noiseless.loc[r]['expectation_values_hamiltonian'])
        true_fidelity = stats_df_noiseless.loc[r]['gs_fidelity']

        data = [r, stats_df_noiseless.loc[r]['E0'], vqe_energy, true_fidelity]

        # stats to compute bounds
        statistics = {'expectation_values': stats_df_noiseless.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df_noiseless.loc[r]['variances_hamiltonian']}

        # compute bounds
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        data += [bounds.lower_bound, bounds.upper_bound]

        plot_df_noiseless.loc[-1] = data
        plot_df_noiseless.index += 1

    plot_df_noiseless.sort_values('r', inplace=True)
    plot_df_noiseless.set_index('r', inplace=True)
    plot_df_noiseless = plot_df_noiseless[plot_df_noiseless.index >= xmin]
    dists_noiseless = plot_df_noiseless.index

    # compute gramian eigenvalue intervals with noise
    for r in stats_df_noisy.index:
        vqe_energy = np.mean(stats_df_noisy.loc[r]['expectation_values_hamiltonian'])
        true_fidelity = stats_df_noisy.loc[r]['gs_fidelity']

        data = [r, stats_df_noisy.loc[r]['E0'], vqe_energy, true_fidelity]

        # stats to compute bounds
        statistics = {'expectation_values': stats_df_noisy.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df_noisy.loc[r]['variances_hamiltonian']}

        # compute bounds
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        data += [bounds.lower_bound, bounds.upper_bound]

        plot_df_noisy.loc[-1] = data
        plot_df_noisy.index += 1

    plot_df_noisy.sort_values('r', inplace=True)
    plot_df_noisy.set_index('r', inplace=True)
    plot_df_noisy = plot_df_noisy[plot_df_noisy.index >= xmin]
    dists_noisy = plot_df_noisy.index

    # ticks
    xticks = np.arange(start=0, stop=max(
        dists_noisy if len(dists_noisy) > len(dists_noiseless) else dists_noiseless) + 0.5 / 2.0, step=0.5)

    xticks = xticks[xticks >= min(
        dists_noisy if len(dists_noisy) > len(dists_noiseless) else dists_noiseless)]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(5, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(dists_noiseless, plot_df_noiseless['exact'], color='black', label=r'$E_0$')

    # noiseless plot
    if len(dists_noiseless) > 0:
        ax1.plot(plot_df_noiseless.index, plot_df_noiseless['vqe_energy'], marker='o', markersize=3, lw=lw,
                 color=vqe_color_noiseless, label=f'{vqe_label} w/o noise')
        ax1.plot(plot_df_noiseless.index, plot_df_noiseless['lower_bounds'], marker='x', ls='--', markersize=3, lw=lw,
                 color=vqe_color_noiseless, label='Lower Bound w/o noise')

    # noisy plot
    if len(dists_noisy) > 0:
        ax1.plot(plot_df_noisy.index, plot_df_noisy['vqe_energy'], marker='o', markersize=3, lw=lw,
                 color=vqe_color_noisy, label=f'{vqe_label} w/ noise')
        ax1.plot(plot_df_noisy.index, plot_df_noisy['lower_bounds'], marker='x', ls='--', markersize=3, lw=lw,
                 color=vqe_color_noisy, label='Lower Bound w/ noise')

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\langle H \rangle$')

    # plot errors
    ax2 = fig.add_subplot(gs[3:4, 0])
    if len(dists_noiseless) > 0:
        ax2.plot(plot_df_noiseless.index, plot_df_noiseless['vqe_energy'] - plot_df_noiseless['exact'], marker='o',
                 markersize=3, lw=lw, color=vqe_color_noiseless)
        ax2.plot(plot_df_noiseless.index, plot_df_noiseless['exact'] - plot_df_noiseless['lower_bounds'], marker='x',
                 markersize=3, lw=lw, ls='--', color=vqe_color_noiseless)

    if len(dists_noisy) > 0:
        ax2.plot(plot_df_noisy.index, plot_df_noisy['vqe_energy'] - plot_df_noisy['exact'], marker='o',
                 markersize=3, lw=lw, color=vqe_color_noisy)
        ax2.plot(plot_df_noisy.index, plot_df_noisy['exact'] - plot_df_noisy['lower_bounds'], marker='x',
                 markersize=3, lw=lw, ls='--', color=vqe_color_noisy)

    loc = plticker.MultipleLocator(base=err_step)
    ax2.yaxis.set_major_locator(loc)

    ax2.set_xticks(xticks)
    ax2.set_ylabel('Error')

    # plot fidelities
    ax3 = fig.add_subplot(gs[4:, 0])
    if len(dists_noiseless) > 0:
        ax3.plot(plot_df_noiseless.index, plot_df_noiseless['fidelity'], marker='o', markersize=3, lw=lw, ls='-',
                 color=vqe_color_noiseless)

    if len(dists_noisy) > 0:
        ax3.plot(plot_df_noisy.index, plot_df_noisy['fidelity'], marker='o', markersize=3, lw=lw, ls='-',
                 color=vqe_color_noisy)

    loc = plticker.MultipleLocator(base=fid_step)
    ax3.yaxis.set_major_locator(loc)
    ax3.set_xticks(xticks)
    ax3.set_ylabel('Fidelity')

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=12, framealpha=0.75)

    # add grid
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fig.align_ylabels([ax1, ax2, ax3])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        save_as = os.path.join('./plots', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)


def interval_plot(results_dir, vqe_label, save_as=None):
    """

    here we make a plot with two subplots containing Energy + Interval / Ground State Fidelity

    """
    stats_df = pd.read_pickle(os.path.join(results_dir, 'statistics.pkl'))

    plot_df = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'fidelity', 'variance', 'lower_bounds', 'upper_bounds'])

    # compute gramian eigenvalue intervals
    for r in stats_df.index:
        vqe_energy = np.mean(stats_df.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df.loc[r]['variances_hamiltonian'])
        true_fidelity = stats_df.loc[r]['gs_fidelity']

        data = [r, stats_df.loc[r]['E0'], vqe_energy, true_fidelity, vqe_variance]

        # stats to compute bounds
        statistics = {'expectation_values': stats_df.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df.loc[r]['variances_hamiltonian']}

        # compute bounds
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        data += [bounds.lower_bound, bounds.upper_bound]

        plot_df.loc[-1] = data
        plot_df.index += 1

    plot_df.sort_values('r', inplace=True)
    plot_df.set_index('r', inplace=True)

    # select subset
    plot_df = plot_df[plot_df.index >= 0.5]
    xticks = np.arange(start=0, stop=max(plot_df.index) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= plot_df.index.min()]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(5, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:4, 0])
    ax1.plot(plot_df.index, plot_df['exact'], color='black', label=r'$E_0$')
    ax1.plot(plot_df.index, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw, color=vqe_color,
             label=f'{vqe_label}')
    ax1.fill_between(plot_df.index, plot_df['lower_bounds'], plot_df['upper_bounds'],
                     label='Robustness Interval', color=interval_color, alpha=0.25, lw=lw)
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\langle H \rangle$')

    # plot fidelity
    ax2 = fig.add_subplot(gs[4:, 0])
    ax2.plot(plot_df.index, plot_df['fidelity'], marker='o', markersize=3, lw=lw, color=vqe_color)
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Fidelity')
    ax2.set_xlabel('Bond Distance (A)')

    # add grid
    ax1.grid()
    ax2.grid()

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=14, framealpha=0.75)

    fig.align_ylabels([ax1, ax2])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        save_as = os.path.join('./plots', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
        pass


# if __name__ == '__main__':
    # # ---  LiH figures
    # interval_plot('../../results2/lih/basis-set-free/hcb=False/upccgsd/noise=1/210706_190140', vqe_label='UpCCGSD',
    #               save_as='interval_lih_upccgsd.pdf')
    # interval_plot('../../results2/lih/basis-set-free/hcb=False/spa/noise=1/210706_185437', vqe_label='SPA',
    #               save_as='interval_lih_spa.pdf')
    #
    # # SPA
    # lower_bounds_plots('../../results2/lih/basis-set-free/hcb=False/spa/noise=1/210706_185437',
    #                    '../../results2/lih/basis-set-free/hcb=False/spa/noise=0/210706_185138', 'SPA',
    #                    err_step=0.025, fid_step=0.05, save_as='lih_spa_bounds.pdf')
    #
    # # UpCCGSD
    # lower_bounds_plots('../../results2/lih/basis-set-free/hcb=False/upccgsd/noise=1/210706_190140',
    #                    '../../results2/lih/basis-set-free/hcb=False/upccgsd/noise=0/210706_185204', 'UpCCGSD',
    #                    err_step=0.1, fid_step=0.2, save_as='lih_upccgsd_bounds.pdf')

    # --- BeH2 figures
    # SPA
    # lower_bounds_plots('../../results2/beh2/basis-set-free/hcb=False/spa/noise=1/210713_193035',
    #                    '../../results2/beh2/basis-set-free/hcb=False/spa/noise=0/210706_193706', 'SPA',
    #                    err_step=0.1, fid_step=0.25, save_as='beh2_spa_bounds.pdf')

    # # UpCCGSD
    # lower_bounds_plots('',
    #                    '../../results2/beh2/basis-set-free/hcb=False/upccgsd/noise=0/210714_161456', 'UpCCGSD',
    #                    err_step=0.1, fid_step=0.25, save_as=None)
