import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import pandas as pd
from tabulate import tabulate

from lib.compute_bounds import compute_bounds_gramian_eigval
from analyze.plot_specs import init_plot_style, GRAMIAN_EIGVAL_COLOR
from analyze.plot_specs import GRAM_EIGVAL_KWARGS, VQE_LINE_KWARGS, LINEWIDTH, FIGSIZE


def make_plot(results_dir, vqe_label, error_limits, mol_img, mol_img_xy, zoom, xmin=0.0, save_as=None):
    stats_df = pd.read_pickle(os.path.join(results_dir, 'hamiltonian_statistics.pkl'))
    plot_df = compute_bounds_gramian_eigval(stats_df, confidence_level=0.01)
    plot_df = plot_df[plot_df.index >= xmin]
    dists = plot_df.index

    print(tabulate(plot_df, headers='keys', tablefmt='presto', floatfmt=".9f"))

    # ticks
    xticks = np.arange(start=0, stop=max(dists) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= min(dists)]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=FIGSIZE)
    gs = fig.add_gridspec(4, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.plot(dists, plot_df['exact'], color='black', label=r'$E_0$', lw=2 * LINEWIDTH)

    ax1.plot(plot_df.index, plot_df['vqe_energy'], **VQE_LINE_KWARGS, label=vqe_label)
    ax1.plot(plot_df.index, plot_df['lower_bound'], **GRAM_EIGVAL_KWARGS, label='RI (Gramian Eigenvalue)')
    ax1.plot(plot_df.index, plot_df['upper_bound'], **GRAM_EIGVAL_KWARGS)
    ax1.fill_between(plot_df.index, plot_df['lower_bound'], plot_df['upper_bound'], color=GRAMIAN_EIGVAL_COLOR,
                     alpha=0.1, lw=LINEWIDTH)

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('energy')
    ax1.tick_params(direction='in', pad=5, which='both')

    # plot errors
    ax2 = fig.add_subplot(gs[2:, 0])
    ax2.plot(plot_df.index, plot_df['vqe_energy'] - plot_df['exact'], **VQE_LINE_KWARGS, label=r'$VQE - E_0$')
    ax2.plot(plot_df.index, plot_df['exact'] - plot_df['lower_bound'], **GRAM_EIGVAL_KWARGS, label='$E_0 - LB$')

    ax2.set_ylim(error_limits)
    ax2.set_yscale('log')
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
    ax2.legend(loc='best', ncol=2, fontsize=12, framealpha=0.75)
    fig.align_ylabels([ax1, ax2])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('final-figures/'):
            os.makedirs('final-figures/')

        save_as = os.path.join('final-figures/', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    init_plot_style()
