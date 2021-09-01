import numpy as np
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from tabulate import tabulate

from lib.robustness_interval import SDPInterval, GramianExpectationBound, GramianEigenvalueInterval

# colors
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


def plot_bound_comparison(results_dir, vqe_label, base_color, mol_img, mol_img_xy, zoom, xmin=0.0, save_as=None):
    """

    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe

    """
    stats_df = pd.read_pickle(os.path.join(results_dir, 'all_statistics.pkl'))
    plot_df = pd.DataFrame(columns=['r', 'exact', 'vqe_energy', 'fidelity',
                                    'eigval_lower_bound', 'gram_lower_bound', 'sdp_lower_bound_grouped'])

    # compute intervals
    for r in stats_df.index:
        vqe_energy = np.mean(stats_df.loc[r]['hamiltonian_expectations'])
        true_fidelity = stats_df.loc[r]['gs_fidelity']

        data = [r, stats_df.loc[r]['E0'], vqe_energy, true_fidelity]

        # stats to compute sdp bounds
        clique_statistics = {'pauliclique_expectations': stats_df.loc[r]['pauliclique_expectations'],
                             'pauliclique_variances': stats_df.loc[r]['pauliclique_variances'],
                             'pauliclique_coeffs': None,
                             'paulicliques': stats_df.loc[r]['paulicliques'],
                             'pauliclique_eigenvalues': stats_df.loc[r]['pauliclique_eigenvalues']}

        pauli_coeffs = stats_df.loc[r]['pauli_coeffs']
        pauli_strings = stats_df.loc[r]['pauli_strings']
        const_terms = [i for i, pstr in enumerate(pauli_strings) if len(pstr) == 0]
        normalization_constant = -np.sum([pauli_coeffs[i] for i in const_terms])
        normalization_constant += np.sum(
            [np.abs(pauli_coeffs[i]) for i in set(range(len(pauli_coeffs))) - set(const_terms)])
        hamiltonian_statistics = {'hamiltonian_expectations': stats_df.loc[r]['hamiltonian_expectations'],
                                  'hamiltonian_variances': stats_df.loc[r]['hamiltonian_variances'],
                                  'normalization_const': normalization_constant}

        # compute bounds
        sdp_bounds_grouped = SDPInterval(statistics=clique_statistics, fidelity=true_fidelity, confidence_level=0.01)
        gramian_expect_bounds = GramianExpectationBound(statistics=clique_statistics, fidelity=true_fidelity,
                                                        confidence_level=0.01, method='cliques')
        gramian_eigval_bounds = GramianEigenvalueInterval(statistics=hamiltonian_statistics, fidelity=true_fidelity,
                                                          confidence_level=0.01)

        data += [gramian_eigval_bounds.lower_bound, gramian_expect_bounds.lower_bound, sdp_bounds_grouped.lower_bound]

        plot_df.loc[-1] = data
        plot_df.index += 1

    print(tabulate(plot_df, headers='keys', tablefmt='presto', floatfmt=".9f"))

    plot_df.sort_values('r', inplace=True)
    plot_df.set_index('r', inplace=True)
    plot_df = plot_df[plot_df.index >= xmin]
    dists = plot_df.index

    # ticks
    xticks = np.arange(start=0, stop=max(dists) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= min(dists)]

    # setup plot
    _ = plt.figure(constrained_layout=True, figsize=figsize)
    ax = plt.gca()

    # plot energy values and bounds
    ax.plot(dists, plot_df['exact'], color='black', lw=2 * lw, label=r'exact')

    # noiseless plot
    ax.plot(dists, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw, color=base_color, label=vqe_label,
            markerfacecolor='none')
    ax.plot(dists, plot_df['eigval_lower_bound'], marker='o', ls='--', markersize=3, lw=lw,
            color=colors[1], label='Gramian Eigenvalue Bound', markerfacecolor='none')
    ax.plot(dists, plot_df['gram_lower_bound'], marker='o', ls='--', markersize=3, lw=lw,
            color=colors[2], label='Gramian Expectation Bound', markerfacecolor='none')
    ax.plot(dists, plot_df['sdp_lower_bound_grouped'], marker='o', ls='--', markersize=3, lw=lw,
            color=colors[3], label='SDP Bound', markerfacecolor='none')

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
    # ax.legend(handles, labels, loc='best', ncol=1, fontsize=12, framealpha=0.75, fancybox=False)
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
    plot_bound_comparison(results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=1/210716_170301',
                          vqe_label='UpCCGSD', base_color=colors[0], xmin=0.0, mol_img='./resources/h2.png',
                          mol_img_xy=(0.45, 0.75), zoom=0.15, save_as='comparison_h2_upccgsd.pdf')

    plot_bound_comparison(results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429',
                          vqe_label='UpCCGSD', base_color=colors[0], mol_img='./resources/lih.png',
                          mol_img_xy=(0.45, 0.75), zoom=0.15, save_as='comparison_lih_upccgsd.pdf')
