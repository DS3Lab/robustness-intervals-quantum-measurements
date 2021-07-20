import numpy as np
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns

from lib.robustness_interval import SDPInterval, GramianExpectationBound, GramianEigenvalueInterval

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
sdp_color = colors[1]
sdp_color_grouped = colors[2]
gram_exp_color = colors[3]
gram_eig_color = colors[4]

lw = 1.0


def plot_bound_comparison(results_dir, vqe_label, fid_step, xmin=1.0, save_as=None, annotate_text=None,
                          annotate_xy=None):
    """

    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe

    """
    pauli_stats_df = pd.read_pickle(os.path.join(results_dir, 'pauli_statistics.pkl'))
    hamiltonian_stats_df = pd.read_pickle(os.path.join(results_dir, 'hamiltonian_statistics.pkl'))

    stats_df = pd.concat([pauli_stats_df, hamiltonian_stats_df[['expectation_values_hamiltonian',
                                                                'variances_hamiltonian']]], axis=1)

    plot_df = pd.DataFrame(columns=['r', 'exact', 'vqe_energy', 'fidelity',
                                    'eigval_lower_bound', 'gram_lower_bound', 'sdp_lower_bound_vanilla',
                                    'sdp_lower_bound_grouped'])

    # compute intervals
    for r in stats_df.index:
        vqe_energy = np.mean(stats_df.loc[r]['expectation_values_hamiltonian'])
        true_fidelity = stats_df.loc[r]['gs_fidelity']

        data = [r, stats_df.loc[r]['E0'], vqe_energy, true_fidelity]

        # stats to compute sdp bounds
        sdp_statistics_grouped = {'expectation_values': stats_df.loc[r]['grouped_pauli_expectations'],
                                  'pauli_strings': stats_df.loc[r]['grouped_pauli_strings'],
                                  'pauli_coeffs': stats_df.loc[r]['grouped_pauli_coeffs'],
                                  'pauli_eigenvalues': stats_df.loc[r]['grouped_pauli_eigenvalues']}

        # compute SDP expectation interval with grouping
        sdp_bounds_grouped = SDPInterval(statistics=sdp_statistics_grouped, fidelity=true_fidelity)

        # stats to compute sdp bounds without grouping
        sdp_statistics_vanilla = {'expectation_values': stats_df.loc[r]['pauli_expectations'],
                                  'pauli_strings': stats_df.loc[r]['pauli_strings'],
                                  'pauli_coeffs': stats_df.loc[r]['pauli_coeffs'],
                                  'pauli_eigenvalues': stats_df.loc[r]['pauli_eigenvalues']}

        # compute SDP expectation interval with grouping
        sdp_bounds_vanilla = SDPInterval(statistics=sdp_statistics_vanilla, fidelity=true_fidelity)

        # stats to compute gramian bounds
        pauli_coeffs = stats_df.loc[r]['pauli_coeffs']
        normalization_constant = pauli_coeffs[0] - np.sum(np.abs(pauli_coeffs[1:]))
        statistics = {'expectation_values': stats_df.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df.loc[r]['variances_hamiltonian']}

        # compute gramian bounds
        gramian_eigval_bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        gramian_expect_bounds = GramianExpectationBound(statistics=statistics, fidelity=true_fidelity,
                                                        normalization_constant=normalization_constant)

        data += [gramian_eigval_bounds.lower_bound, gramian_expect_bounds.lower_bound, sdp_bounds_vanilla.lower_bound,
                 sdp_bounds_grouped.lower_bound]

        plot_df.loc[-1] = data
        plot_df.index += 1

    plot_df.sort_values('r', inplace=True)
    plot_df.set_index('r', inplace=True)
    plot_df = plot_df[plot_df.index >= xmin]
    dists = plot_df.index

    # ticks
    xticks = np.arange(start=0, stop=max(dists) + 0.5 / 2.0, step=0.5)
    xticks = xticks[xticks >= min(dists)]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(6, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:5, 0])
    ax1.plot(dists, plot_df['exact'], color='black', label=r'$E_0$')

    # noiseless plot
    ax1.plot(dists, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw, color=vqe_color, label=vqe_label)
    ax1.plot(dists, plot_df['eigval_lower_bound'], marker='x', ls='--', markersize=3, lw=lw,
             color=gram_eig_color, label='Gramian Eigenvalue Bound')
    ax1.plot(dists, plot_df['gram_lower_bound'], marker='x', ls='--', markersize=3, lw=lw,
             color=gram_exp_color, label='Gramian Expectation Bound')
    ax1.plot(dists, plot_df['sdp_lower_bound_vanilla'], marker='x', ls='--', markersize=3, lw=lw,
             color=sdp_color, label='SDP Bound')
    ax1.plot(dists, plot_df['sdp_lower_bound_grouped'], marker='x', ls='--', markersize=3, lw=lw,
             color=sdp_color_grouped, label='SDP Bound + Grouping')

    # annotate
    if annotate_text is not None:
        annotate_xy = (0.1, 0.9) if annotate_xy is None else annotate_xy
        ax1.annotate(text=annotate_text, xycoords='axes fraction', xy=annotate_xy, horizontalalignment='left',
                     verticalalignment='top', fontsize=28)

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\langle H \rangle$')

    # plot fidelities
    ax2 = fig.add_subplot(gs[5:, 0])
    ax2.plot(plot_df.index, plot_df['fidelity'], marker='o', markersize=3, lw=lw, ls='-', color=vqe_color)

    loc = plticker.MultipleLocator(base=fid_step)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Fidelity')

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=12, framealpha=0.75)

    # add grid
    ax1.grid()
    ax2.grid()

    fig.align_ylabels([ax1, ax2])

    if save_as is not None:
        if not os.path.exists('figures/bound_comparison'):
            os.makedirs('figures/bound_comparison')

        save_as = os.path.join('figures/bound_comparison', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
        return

    plt.show()


if __name__ == '__main__':
    # noisy plots
    plot_bound_comparison(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=1/210716_170002',
                          vqe_label='SPA', xmin=0.0, fid_step=0.01, save_as='h2_comparison_spa_noisy.pdf',
                          annotate_text=r'H$_2$', annotate_xy=(0.2, 0.95))

    plot_bound_comparison(results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=1/210716_170301',
                          vqe_label='UpCCGSD', xmin=0.0, fid_step=0.025, save_as='h2_comparison_upccgsd_noisy.pdf',
                          annotate_text=r'H$_2$', annotate_xy=(0.2, 0.95))

    plot_bound_comparison(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=1/210716_162531',
                          vqe_label='SPA', fid_step=0.05, save_as='lih_comparison_spa_noisy.pdf',
                          annotate_text=r'LiH', annotate_xy=(0.2, 0.95))

    plot_bound_comparison(results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429',
                          vqe_label='UpCCGSD', fid_step=0.05, save_as='lih_comparison_upccgsd_noisy.pdf',
                          annotate_text=r'LiH', annotate_xy=(0.2, 0.95))

    plot_bound_comparison(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=1/210716_171529',
                          vqe_label='SPA', fid_step=0.2, save_as='beh2_comparison_spa_noisy.pdf',
                          annotate_text=r'BeH$_2$', annotate_xy=(0.75, 0.8))

    # noiseless plots
    plot_bound_comparison(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=0/210716_162509',
                          vqe_label='SPA', xmin=0.0, fid_step=1e-7, save_as='h2_comparison_spa_no_noise.pdf',
                          annotate_text=r'H$_2$', annotate_xy=(0.8, 0.2))

    plot_bound_comparison(results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=0/210716_162519',
                          vqe_label='UpCCGSD', xmin=0.0, fid_step=5e-7, save_as='h2_comparison_upccgsd_no_noise.pdf',
                          annotate_text=r'H$_2$', annotate_xy=(0.8, 0.2))

    plot_bound_comparison(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
                          vqe_label='SPA', fid_step=0.05, save_as='lih_comparison_spa_no_noise.pdf',
                          annotate_text=r'LiH')

    plot_bound_comparison(results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=0/210716_162453',
                          vqe_label='UpCCGSD', fid_step=5e-7, save_as='lih_comparison_upccgsd_no_noise.pdf',
                          annotate_text=r'LiH')

    plot_bound_comparison(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210716_171413',
                          vqe_label='SPA', fid_step=0.2, save_as='beh2_comparison_spa_no_noise.pdf',
                          annotate_text=r'BeH$_2$', annotate_xy=(0.1, 0.7))
