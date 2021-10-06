import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import pandas as pd

from lib.compute_bounds import compute_bounds_gramian_eigval
from analyze.plot_specs import init_plot_style
from analyze.plot_specs import GRAM_EIGVAL_KWARGS, VQE_LINE_KWARGS, LINEWIDTH, FIGSIZE, GRAMIAN_EIGVAL_COLOR


def interval_plot(results_dir, vqe_label, save=False):
    """
    here we make a plot with two subplots containing Energy + Interval / Ground state fidelity
    """
    stats_df = pd.read_pickle(os.path.join(results_dir, 'hamiltonian_statistics.pkl'))
    plot_df = compute_bounds_gramian_eigval(stats_df, confidence_level=0.01)

    # select subset
    plot_df = plot_df[plot_df.index >= 0.5]
    xticks = np.arange(start=0, stop=max(plot_df.index) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= plot_df.index.min()]

    # setup plot
    plt.figure(constrained_layout=True, figsize=FIGSIZE)
    ax = plt.gca()

    # plot energy values and bounds
    ax.plot(plot_df.index, plot_df['exact'], color='black', lw=2 * LINEWIDTH, label=r'exact')
    ax.plot(plot_df.index, plot_df['vqe_energy'], label=vqe_label, **VQE_LINE_KWARGS)
    ax.fill_between(plot_df.index, plot_df['lower_bound'], plot_df['upper_bound'], color=GRAMIAN_EIGVAL_COLOR,
                    alpha=0.1, lw=LINEWIDTH)
    ax.plot(plot_df.index, plot_df['lower_bound'], **GRAM_EIGVAL_KWARGS, label='RI (Gramian Eigenvalue)')
    ax.plot(plot_df.index, plot_df['upper_bound'], **GRAM_EIGVAL_KWARGS)
    ax.set_xticks(xticks)
    ax.tick_params(direction='in', pad=5, which='both')
    ax.set_ylabel(r'Energy')
    ax.set_xlabel(r'Bond Distance ($\AA$)')

    # put in molecule
    img = mpimg.imread('./resources/lih.png')
    imagebox = OffsetImage(img, zoom=0.15, resample=False)
    ab = AnnotationBbox(imagebox, xy=(0.65, 0.65), xycoords='figure fraction', frameon=False)
    ax.add_artist(ab)
    plt.draw()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', fancybox=False, ncol=1, fontsize=14, framealpha=0.75)

    if save:
        if not os.path.exists('./final-figures'):
            os.makedirs('./final-figures/')

        save_as = os.path.join('./final-figures', 'lih_gramian_interval.pdf')
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


if __name__ == '__main__':
    init_plot_style()
