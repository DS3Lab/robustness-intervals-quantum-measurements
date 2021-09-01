import numpy as np
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from lib.robustness_interval import GramianEigenvalueInterval

font_size = 12
font_size_large = 16
sns.set_style('ticks')
colors = sns.color_palette('muted')
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = font_size_large
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["axes.linewidth"] = 1.
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['ytick.minor.size'] = 2

lw = 1.0
figsize = (6, 4.5)


def make_plot(results_dir, vqe_label, base_color, error_limits, mol_img, mol_img_xy, zoom, xmin=0.0, save_as=None):
    try:
        stats_df = pd.read_pickle(os.path.join(results_dir, 'all_statistics.pkl'))
    except FileNotFoundError:
        stats_df = None

    if stats_df is None:
        stats_df = pd.read_pickle(os.path.join(results_dir, 'hamiltonian_statistics.pkl'))

    plot_df = pd.DataFrame(columns=['r', 'exact', 'vqe_energy', 'fidelity', 'lower_bounds', 'upper_bounds'])

    # compute gramian eigenvalue intervals without noise
    for r in stats_df.index:
        vqe_energy = np.mean(stats_df.loc[r]['hamiltonian_expectations'])
        true_fidelity = stats_df.loc[r]['gs_fidelity']

        data = [r, stats_df.loc[r]['E0'], vqe_energy, true_fidelity]

        # stats to compute bounds
        statistics = {'hamiltonian_expectations': stats_df.loc[r]['hamiltonian_expectations'],
                      'hamiltonian_variances': stats_df.loc[r]['hamiltonian_variances']}

        # compute bounds
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity, confidence_level=0.01)
        data += [bounds.lower_bound, bounds.upper_bound]

        plot_df.loc[-1] = data
        plot_df.index += 1

    plot_df = plot_df.sort_values('r').set_index('r')
    plot_df = plot_df[plot_df.index >= xmin]
    dists = plot_df.index

    # ticks
    xticks = np.arange(start=0, stop=max(dists) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= min(dists)]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(4, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.plot(dists, plot_df['exact'], color='black', label=r'exact', lw=2 * lw)

    # noiseless plot
    ax1.plot(plot_df.index, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw,
             color=base_color, label=vqe_label, markerfacecolor='none')
    ax1.plot(plot_df.index, plot_df['lower_bounds'], marker='o', ls='--', markersize=3, lw=lw,
             color=colors[1], label='Lower Bound', markerfacecolor='none')

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('energy')
    ax1.tick_params(direction='in', pad=5, which='both')

    # plot errors
    ax2 = fig.add_subplot(gs[2:, 0])
    ax2.plot(plot_df.index, plot_df['vqe_energy'] - plot_df['exact'], marker='o', markersize=3, lw=lw, color=base_color,
             markerfacecolor='none')
    ax2.plot(plot_df.index, plot_df['exact'] - plot_df['lower_bounds'], marker='o', markersize=3, lw=lw, ls='--',
             color=colors[1], markerfacecolor='none')

    ax2.set_ylim(error_limits)
    ax2.set_yscale('log')
    # ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Error')
    ax2.tick_params(direction='in', pad=5, which='both')
    ax2.set_xlabel(r'Bond Distance ($\AA$)')

    # put in molecule
    img = mpimg.imread(mol_img)
    imagebox = OffsetImage(img, zoom=zoom, resample=False)
    ab = AnnotationBbox(imagebox, xy=mol_img_xy, xycoords='figure fraction', frameon=False)
    ax2.add_artist(ab)
    plt.draw()

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=12, framealpha=0.75)
    fig.align_ylabels([ax1, ax2])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('final-figures/'):
            os.makedirs('final-figures/')

        save_as = os.path.join('final-figures/', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    make_plot('../results/h2/basis-set-free/hcb=False/spa/noise=1/210716_170002',
              vqe_label='SPA', base_color=colors[0], error_limits=(1e-3, 0.1),
              mol_img='resources/h2.png', mol_img_xy=(0.75, 0.65), zoom=0.125, save_as='h2_noisy_spa.pdf')
    make_plot('../results/h2/basis-set-free/hcb=False/upccgsd/noise=1/210716_170301',
              vqe_label='UpCCGSD', base_color=colors[0], error_limits=(1e-2, 1.0),
              mol_img='resources/h2.png', mol_img_xy=(0.75, 0.6125),
              zoom=0.125, save_as='h2_noisy_upccgsd.pdf')

    # make_plot('../results/lih/basis-set-free/hcb=False/upccgsd/noise=0/210716_162453',
    #           vqe_label='UpCCGSD', base_color=colors[0], error_limits=(1e-11, 1e-6),
    #           mol_img='resources/lih.png', mol_img_xy=(0.75, 0.65),
    #           zoom=0.125, save_as='lih_noiseless_upccgsd.pdf')
    # make_plot('../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
    #           vqe_label='SPA', base_color=colors[0], error_limits=(1e-14, 0.1),
    #           mol_img='resources/lih.png', mol_img_xy=(0.75, 0.65), zoom=0.125, save_as='lih_noiseless_spa.pdf')
    #
    # make_plot('../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429',
    #           vqe_label='UpCCGSD', base_color=colors[0], error_limits=(1e-2, 0.2),
    #           mol_img='resources/lih.png', mol_img_xy=(0.4, 0.8375), zoom=0.125, save_as='lih_noisy_upccgsd.pdf')
    # make_plot('../results/lih/basis-set-free/hcb=False/spa/noise=1/210716_162531',
    #           vqe_label='SPA', base_color=colors[0], error_limits=(1e-3, 0.1),
    #           mol_img='resources/lih.png', mol_img_xy=(0.75, 0.65), zoom=0.125, save_as='lih_noisy_spa.pdf')

    # make_plot('../results/beh2/basis-set-free/hcb=False/upccgsd/noise=0/210716_171432',
    #           vqe_label='UpCCGSD', base_color=colors[0], error_limits=(1e-4, 0.1),
    #           mol_img='resources/beh2.png', mol_img_xy=(0.34, 0.8),
    #           zoom=0.125)  # , save_as='beh2_noiseless_upccgsd.pdf')
    # make_plot('../results/beh2/basis-set-free/hcb=False/spa/noise=0/210716_171413',
    #           vqe_label='SPA', base_color=colors[0], error_limits=(1e-5, 0.3),
    #           mol_img='resources/beh2.png', mol_img_xy=(0.34, 0.8), zoom=0.125, save_as='beh2_noiseless_spa.pdf')
    # make_plot('../results/beh2/basis-set-free/hcb=False/spa/noise=1/210716_171529',
    #           vqe_label='SPA', base_color=colors[0], error_limits=(1e-3, 1.0),
    #           mol_img='resources/beh2.png', mol_img_xy=(0.34, 0.8), zoom=0.125)  # , save_as='beh2_noisy_spa.pdf')

#
