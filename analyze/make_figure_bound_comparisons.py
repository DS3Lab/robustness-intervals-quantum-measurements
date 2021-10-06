import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import pandas as pd
from tabulate import tabulate

from lib.compute_bounds import compute_bounds_gramian_eigval, compute_bounds_sdp, compute_bounds_gramian_expectation
from lib.compute_bounds import CLIQUE_PARTITION
from analyze.plot_specs import init_plot_style, GRAMIAN_EIGVAL_COLOR
from analyze.plot_specs import GRAM_EIGVAL_KWARGS, VQE_LINE_KWARGS, LINEWIDTH, FIGSIZE, GRAM_EXPEC_KWARGS, SDP_KWARGS


def plot_bound_comparison(results_dir, vqe_label, mol_img, mol_img_xy, zoom, xmin=0.0, save_as=None):
    """
    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe
    """
    stats_df = pd.read_pickle(os.path.join(results_dir, 'all_statistics.pkl'))
    df1 = compute_bounds_gramian_eigval(stats_df, confidence_level=0.01)
    df2 = compute_bounds_sdp(stats_df, confidence_level=0.01, partition=CLIQUE_PARTITION)
    df3 = compute_bounds_gramian_expectation(stats_df, confidence_level=0.01, partition=CLIQUE_PARTITION)

    df1 = df1.rename(columns={'lower_bound': 'gram_eigval_lower_bound', 'upper_bound': 'gram_eigval_upper_bound',
                              'vqe_energy': 'gram_eigval_energy'})
    df2 = df2.rename(columns={'lower_bound': 'sdp_lower_bound', 'upper_bound': 'sdp_upper_bound',
                              'vqe_energy': 'sdp_energy'})
    df3 = df3.rename(columns={'lower_bound': 'gram_expectation_lower_bound',
                              'upper_bound': 'gram_expectation_upper_bound',
                              'vqe_energy': 'gram_expectation_energy'})

    plot_df = pd.merge(pd.merge(df1, df2, on=['r', 'exact', 'fidelity']), df3, on=['r', 'exact', 'fidelity'])
    print(tabulate(plot_df, headers='keys', tablefmt='presto', floatfmt=".9f"))

    plot_df = plot_df[plot_df.index >= xmin]
    dists = plot_df.index

    # ticks
    xticks = np.arange(start=0, stop=max(dists) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= min(dists)]

    # setup plot
    _ = plt.figure(constrained_layout=True, figsize=FIGSIZE)
    ax = plt.gca()

    # plot energy values and bounds
    ax.plot(dists, plot_df['exact'], color='black', lw=2 * LINEWIDTH, label=r'exact')

    # noiseless plot
    ax.plot(dists, plot_df['gram_eigval_energy'], **VQE_LINE_KWARGS, label=vqe_label)

    ax.plot(dists, plot_df['gram_eigval_lower_bound'], **GRAM_EIGVAL_KWARGS)
    ax.plot(dists, plot_df['gram_eigval_upper_bound'], **GRAM_EIGVAL_KWARGS, label='RI (Gramian Eigenvalue)')
    ax.fill_between(plot_df.index, plot_df['gram_eigval_lower_bound'], plot_df['gram_eigval_upper_bound'],
                    color=GRAMIAN_EIGVAL_COLOR, alpha=0.1)

    ax.plot(dists, plot_df['gram_expectation_lower_bound'], **GRAM_EXPEC_KWARGS, label='LB (Gramian Expectation)')

    ax.plot(dists, plot_df['sdp_lower_bound'], **SDP_KWARGS, label='RI (SDP)')
    ax.plot(dists, plot_df['sdp_upper_bound'], **SDP_KWARGS)

    ax.set_xticks(xticks)
    ax.tick_params(direction='in', pad=5, which='both')
    ax.set_ylabel(r'Energy')
    ax.set_xlabel(r'Bond Distance ($\AA$)')

    # put in molecule
    img = mpimg.imread(mol_img)
    imagebox = OffsetImage(img, zoom=zoom, resample=False)
    ab = AnnotationBbox(imagebox, xy=mol_img_xy, xycoords='figure fraction', frameon=False)
    ax.add_artist(ab)
    plt.draw()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=12, framealpha=0.75,
              fancybox=False)

    if save_as is not None:
        if not os.path.exists('final-figures/'):
            os.makedirs('final-figures/')

        save_as = os.path.join('final-figures/', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
        return

    plt.show()


if __name__ == '__main__':
    init_plot_style()
