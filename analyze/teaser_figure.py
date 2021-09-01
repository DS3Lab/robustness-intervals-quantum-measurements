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

lw = 1.0
figsize = (6, 4.32)


def interval_plot(results_dir, vqe_label, base_color, save=False):
    """

    here we make a plot with two subplots containing Energy + Interval / Ground State Fidelity

    """
    stats_df = pd.read_pickle(os.path.join(results_dir, 'hamiltonian_statistics.pkl'))

    plot_df = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'fidelity', 'variance', 'lower_bounds', 'upper_bounds'])

    # compute gramian eigenvalue intervals
    for r in stats_df.index:
        vqe_energy = np.mean(stats_df.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df.loc[r]['variances_hamiltonian'])
        true_fidelity = stats_df.loc[r]['gs_fidelity']

        data = [r, stats_df.loc[r]['E0'], vqe_energy, true_fidelity, vqe_variance]

        # stats to compute bounds
        statistics = {'hamiltonian_expectations': stats_df.loc[r]['expectation_values_hamiltonian'],
                      'hamiltonian_variances': stats_df.loc[r]['variances_hamiltonian']}

        # compute bounds
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity, confidence_level=0.01)
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
    plt.figure(constrained_layout=True, figsize=figsize)
    ax = plt.gca()

    # plot energy values and bounds
    ax.plot(plot_df.index, plot_df['exact'], color='black', lw=2 * lw, label=r'exact')
    ax.plot(plot_df.index, plot_df['vqe_energy'], marker='o', linestyle='--', markersize=5, lw=lw, color=base_color,
            label=vqe_label, markerfacecolor='none')
    ax.fill_between(plot_df.index, plot_df['lower_bounds'], plot_df['upper_bounds'],
                    label='Robustness Interval', color=base_color, alpha=0.1, lw=lw)
    ax.plot(plot_df.index, plot_df['lower_bounds'], lw=lw, color=base_color,
            alpha=0.2)
    ax.plot(plot_df.index, plot_df['upper_bounds'], lw=lw, color=base_color,
            alpha=0.2)
    ax.set_xticks(xticks)
    ax.tick_params(direction='in', pad=5, which='both')
    ax.set_ylabel(r'Energy')
    ax.set_xlabel(r'Bond Distance ($\AA$)')

    # put in molecule
    img = mpimg.imread('./resources/lih.png')
    imagebox = OffsetImage(img, zoom=0.15, resample=False)
    ab = AnnotationBbox(imagebox, xy=(0.4, 0.8), xycoords='figure fraction', frameon=False)
    ax.add_artist(ab)
    plt.draw()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', fancybox=False, ncol=1, fontsize=14, framealpha=0.75)

    if save:
        if not os.path.exists('./final-figures'):
            os.makedirs('./final-figures/')

        save_as = os.path.join('./final-figures', 'lih_bounds.pdf')
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


if __name__ == '__main__':
    interval_plot('../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429', base_color=colors[0],
                  vqe_label='VQE', save=True)
