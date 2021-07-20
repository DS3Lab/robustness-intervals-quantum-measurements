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

color_true_f_bound = colors[1]
color_eckart_bound = colors[2]
color_weinstein_bound = colors[3]
color_mcclean_bound = colors[4]
color_our_bound = colors[5]
gap_color = colors[6]

lw = 1.0


def plot_lower_bounds_different_fidelities(results_dir, vqe_label, fid_step, var_step, xmin=1.0, save_as=None):
    """

    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe

    """
    fp = os.path.join(results_dir, 'hamiltonian_statistics.pkl')
    stats_df_noiseless = pd.read_pickle(fp)

    plot_df = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'vqe_variance', 'gap',
                 'true_fidelity', 'eckart_fidelity', 'weinstein_fidelity', 'mcclean_fidelity', 'our_fidelity',
                 'bound_true_f', 'bound_eckart_f', 'bound_weinstein_f', 'bound_mcclean_f', 'bound_our_f'])

    # compute gramian eigenvalue intervals without noise
    for r in stats_df_noiseless.index:
        vqe_energy = np.mean(stats_df_noiseless.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df_noiseless.loc[r]['variances_hamiltonian'])

        true_fidelity = stats_df_noiseless.loc[r]['gs_fidelity']
        e0 = stats_df_noiseless.loc[r]['E0']
        e1 = stats_df_noiseless.loc[r]['E1']

        eckart_fidelity = max(0, (e1 - vqe_energy) / (e1 - e0))
        weinstein_fidelity = 0.5 if vqe_energy <= 0.5 * (e0 + e1) else 0.0

        if vqe_energy <= 0.5 * (e0 + e1):
            mcclean_fidelity = max(0, 1 - np.sqrt(vqe_variance) / (e1 - e0))
        else:
            mcclean_fidelity = 0.0

        if vqe_energy <= 0.5 * (e0 + e1) and np.sqrt(vqe_variance) <= 0.5 * (e1 - e0):
            our_fidelity = 0.5 + 0.5 * np.sqrt(1 - 4 * vqe_variance / ((e1 - e0) ** 2))
        else:
            our_fidelity = 0.0

        data = [r, stats_df_noiseless.loc[r]['E0'], vqe_energy, vqe_variance, e1 - e0,
                true_fidelity, eckart_fidelity, weinstein_fidelity, mcclean_fidelity, our_fidelity, ]

        # stats to compute bounds
        statistics = {'expectation_values': stats_df_noiseless.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df_noiseless.loc[r]['variances_hamiltonian']}

        # compute bounds based on true fidelity
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Eckart fidelity
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=eckart_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Weinstein
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=weinstein_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Mcclean
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=mcclean_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Our method
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=our_fidelity)
        data += [bounds.lower_bound]

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
    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(5, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(dists, plot_df['exact'], color='black', label=r'$E_0$')

    # plot bounds
    ax1.plot(dists, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw, color=vqe_color, label=vqe_label)
    ax1.plot(dists, plot_df['bound_true_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_true_f_bound,
             label='True Fidelity')
    ax1.plot(dists, plot_df['bound_eckart_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_eckart_bound,
             label='Eckart Fidelity Bound')
    ax1.plot(dists, plot_df['bound_weinstein_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_weinstein_bound,
             label='Weinstein Bound')
    ax1.plot(dists, plot_df['bound_mcclean_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_mcclean_bound,
             label='McClean Fidelity bound')
    ax1.plot(dists, plot_df['bound_our_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_our_bound,
             label='Our Fidelity bound')

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\langle H \rangle$')

    # plot fidelities
    ax2 = fig.add_subplot(gs[3:4, 0])

    ax2.plot(dists, plot_df['true_fidelity'], marker='x', markersize=3, lw=lw, ls='-', color=color_true_f_bound)
    ax2.plot(plot_df[plot_df['eckart_fidelity'] > 0].index, plot_df[plot_df['eckart_fidelity'] > 0]['eckart_fidelity'],
             marker='x', markersize=3, lw=lw, ls='-', color=color_eckart_bound)
    ax2.plot(plot_df[plot_df['weinstein_fidelity'] > 0].index,
             plot_df[plot_df['weinstein_fidelity'] > 0]['weinstein_fidelity'], marker='x', markersize=3, lw=lw, ls='-',
             color=color_weinstein_bound)
    ax2.plot(plot_df[plot_df['mcclean_fidelity'] > 0].index,
             plot_df[plot_df['mcclean_fidelity'] > 0]['mcclean_fidelity'], marker='x', markersize=3, lw=lw, ls='-',
             color=color_mcclean_bound)
    ax2.plot(plot_df[plot_df['our_fidelity'] > 0].index, plot_df[plot_df['our_fidelity'] > 0]['our_fidelity'],
             marker='x', markersize=3, lw=lw, ls='-', color=color_our_bound)

    loc = plticker.MultipleLocator(base=fid_step)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Fidelity')

    # plot variance and gap
    ax3 = fig.add_subplot(gs[4:, 0])

    ax3.plot(dists, plot_df['vqe_variance'].apply(np.sqrt), marker='x', markersize=3, lw=lw, ls='-', color=vqe_color,
             label=r'$\Delta H_\rho$')

    ax3.set_ylim(-0.01, 0.5)
    loc = plticker.MultipleLocator(base=var_step)
    ax3.yaxis.set_major_locator(loc)
    ax3.set_xticks(xticks)
    ax3.set_ylabel(r'$\Delta H_\rho$')

    ax4 = ax3.twinx()
    ax4.set_ylim(-0.01, 0.5)
    loc = plticker.MultipleLocator(base=var_step)
    ax4.yaxis.set_major_locator(loc)
    ax4.plot(plot_df.index, plot_df['gap'], color=gap_color, lw=lw, ls='--', label=r'$\delta = E_1 - E_0$')
    ax4.set_ylabel(r'$E_1 - E_0$')

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=10, framealpha=0.5)

    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(handles3 + handles4, labels3 + labels4, loc='upper center', ncol=1, fontsize=10, framealpha=0.5)

    # add grid
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fig.align_ylabels([ax1, ax2, ax3, ax4])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('figures/eigenval_fidelity_bounds'):
            os.makedirs('figures/eigenval_fidelity_bounds')

        save_as = os.path.join('figures/eigenval_fidelity_bounds', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)


def plot_lower_bounds_different_fidelities2(results_dir, vqe_label, fid_step, var_step, xmin=1.0, save_as=None):
    """

    here we make a plot with three subplots containing Energy + Lower Bounds / Error / Ground State Fidelity,
    for noisy and noiseless vqe

    """
    fp = os.path.join(results_dir, 'hamiltonian_statistics.pkl')
    stats_df_noiseless = pd.read_pickle(fp)

    plot_df = pd.DataFrame(
        columns=['r', 'exact', 'vqe_energy', 'vqe_variance', 'gap',
                 'true_fidelity', 'eckart_fidelity', 'weinstein_fidelity', 'mcclean_fidelity', 'our_fidelity',
                 'var_fid',
                 'bound_true_f', 'bound_eckart_f', 'bound_weinstein_f', 'bound_mcclean_f', 'bound_our_f',
                 'bound_var_fid'])

    # compute gramian eigenvalue intervals without noise
    for r in stats_df_noiseless.index:
        vqe_energy = np.mean(stats_df_noiseless.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df_noiseless.loc[r]['variances_hamiltonian'])

        true_fidelity = stats_df_noiseless.loc[r]['gs_fidelity']
        e0 = stats_df_noiseless.loc[r]['E0']
        e1 = stats_df_noiseless.loc[r]['E1']

        if vqe_energy ** 2 >= 1:
            variance_fidelity = 1.0 / (1.0 + vqe_variance)
        else:
            variance_fidelity = 0

        eckart_fidelity = max(0, (e1 - vqe_energy) / (e1 - e0))
        weinstein_fidelity = 0.5 if vqe_energy <= 0.5 * (e0 + e1) else 0.0

        if vqe_energy <= 0.5 * (e0 + e1):
            mcclean_fidelity = max(0, 1 - np.sqrt(vqe_variance) / (e1 - e0))
        else:
            mcclean_fidelity = 0.0

        if vqe_energy <= 0.5 * (e0 + e1) and np.sqrt(vqe_variance) <= 0.5 * (e1 - e0):
            our_fidelity = 0.5 + 0.5 * np.sqrt(1 - 4 * vqe_variance / ((e1 - e0) ** 2))
        else:
            our_fidelity = 0.0

        data = [r, stats_df_noiseless.loc[r]['E0'], vqe_energy, vqe_variance, e1 - e0,
                true_fidelity, eckart_fidelity, weinstein_fidelity, mcclean_fidelity, our_fidelity, variance_fidelity]

        # stats to compute bounds
        statistics = {'expectation_values': stats_df_noiseless.loc[r]['expectation_values_hamiltonian'],
                      'variances': stats_df_noiseless.loc[r]['variances_hamiltonian']}

        # compute bounds based on true fidelity
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=true_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Eckart fidelity
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=eckart_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Weinstein
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=weinstein_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Mcclean
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=mcclean_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on Our method
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=our_fidelity)
        data += [bounds.lower_bound]

        # compute bounds based on variance ~ fidelity
        bounds = GramianEigenvalueInterval(statistics=statistics, fidelity=variance_fidelity)
        data += [bounds.lower_bound]

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
    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    gs = fig.add_gridspec(10, 2)

    # plot energy values and bounds
    ax1 = fig.add_subplot(gs[:7, 0])
    ax1.plot(dists, plot_df['exact'], color='black', label=r'$E_0$')

    ax1.plot(dists, plot_df['vqe_energy'], marker='o', markersize=3, lw=lw, color=vqe_color, label=vqe_label)
    # ax1.plot(dists, plot_df['bound_true_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_true_f_bound,
    #          label='True Fidelity')
    # ax1.plot(dists, plot_df['bound_eckart_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_eckart_bound,
    #          label='Eckart Fidelity Bound')
    # ax1.plot(dists, plot_df['bound_weinstein_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_weinstein_bound,
    #          label='Weinstein Bound')
    # ax1.plot(dists, plot_df['bound_mcclean_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_mcclean_bound,
    #          label='McClean Fidelity bound')
    # ax1.plot(dists, plot_df['bound_our_f'], marker='x', ls='--', markersize=3, lw=lw, color=color_our_bound,
    #          label='Our Fidelity bound')

    ax1.plot(dists, plot_df['bound_var_fid'], marker='x', ls='--', markersize=3, lw=lw, color=color_our_bound,
             label='Variance ~ Fidelity bound')

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\langle H \rangle$')

    # plot fidelities
    ax2 = fig.add_subplot(gs[7:, 0])

    ax2.plot(dists, plot_df['true_fidelity'], marker='x', markersize=3, lw=lw, ls='-', color=color_true_f_bound)
    # ax2.plot(plot_df[plot_df['eckart_fidelity'] > 0].index, plot_df[plot_df['eckart_fidelity'] > 0]['eckart_fidelity'],
    #          marker='x', markersize=3, lw=lw, ls='-', color=color_eckart_bound)
    # ax2.plot(plot_df[plot_df['weinstein_fidelity'] > 0].index,
    #          plot_df[plot_df['weinstein_fidelity'] > 0]['weinstein_fidelity'], marker='x', markersize=3, lw=lw, ls='-',
    #          color=color_weinstein_bound)
    # ax2.plot(plot_df[plot_df['mcclean_fidelity'] > 0].index,
    #          plot_df[plot_df['mcclean_fidelity'] > 0]['mcclean_fidelity'], marker='x', markersize=3, lw=lw, ls='-',
    #          color=color_mcclean_bound)
    # ax2.plot(plot_df[plot_df['our_fidelity'] > 0].index, plot_df[plot_df['our_fidelity'] > 0]['our_fidelity'],
    #          marker='x', markersize=3, lw=lw, ls='-', color=color_our_bound)
    ax2.plot(plot_df[plot_df['var_fid'] > 0].index, plot_df[plot_df['var_fid'] > 0]['var_fid'],
             marker='x', markersize=3, lw=lw, ls='-', color=color_our_bound)

    loc = plticker.MultipleLocator(base=fid_step)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Fidelity')

    # plot energy values and bounds
    ax5 = fig.add_subplot(gs[:6, 1])

    ax5.plot(dists, plot_df['vqe_energy'] - plot_df['exact'], marker='o', markersize=3, lw=lw, color=vqe_color,
             label=vqe_label)
    # ax5.plot(dists, plot_df['exact'] - plot_df['bound_true_f'], marker='x', ls='--', markersize=3, lw=lw,
    #          color=color_true_f_bound, label='True Fidelity')
    # ax5.plot(dists, plot_df['exact'] - plot_df['bound_eckart_f'], marker='x', ls='--', markersize=3, lw=lw,
    #          color=color_eckart_bound, label='Eckart Fidelity Bound')
    # ax5.plot(dists, plot_df['exact'] - plot_df['bound_weinstein_f'], marker='x', ls='--', markersize=3, lw=lw,
    #          color=color_weinstein_bound, label='Weinstein Bound')
    # ax5.plot(dists, plot_df['exact'] - plot_df['bound_mcclean_f'], marker='x', ls='--', markersize=3, lw=lw,
    #          color=color_mcclean_bound, label='McClean Fidelity bound')
    # ax5.plot(dists, plot_df['exact'] - plot_df['bound_our_f'], marker='x', ls='--', markersize=3, lw=lw,
    #          color=color_our_bound, label='Our Fidelity bound')
    ax5.plot(dists, plot_df['exact'] - plot_df['bound_var_fid'], marker='x', ls='--', markersize=3, lw=lw,
             color=color_our_bound, label='Variance ~ Fidelity bound')

    ax5.set_xticks(xticks)
    ax5.xaxis.set_ticklabels([])
    ax5.set_ylabel(r'Error')
    ax5.set_ylim(-0.03, 0.06)

    # plot variance and gap
    ax3 = fig.add_subplot(gs[6:, 1])

    ax3.plot(dists, plot_df['vqe_variance'].apply(np.sqrt), marker='x', markersize=3, lw=lw, ls='-', color=vqe_color,
             label=r'$\Delta H_\rho$')

    ax3.set_ylim(-0.01, 0.6)
    loc = plticker.MultipleLocator(base=var_step)
    ax3.yaxis.set_major_locator(loc)
    ax3.set_xticks(xticks)
    ax3.set_ylabel(r'$\Delta H_\rho$')

    ax4 = ax3.twinx()
    ax4.set_ylim(-0.01, 0.6)
    loc = plticker.MultipleLocator(base=var_step)
    ax4.yaxis.set_major_locator(loc)
    ax4.plot(plot_df.index, plot_df['gap'], color=gap_color, lw=lw, ls='--', label=r'$\delta = E_1 - E_0$')
    ax4.set_ylabel(r'$E_1 - E_0$')

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=10, framealpha=0.5)

    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(handles3 + handles4, labels3 + labels4, loc='upper center', ncol=1, fontsize=10, framealpha=0.5)

    # gs.update(wspace=0.01, hspace=0.01)

    # add grid
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax5.grid()

    fig.align_ylabels([ax1, ax2, ax3, ax4])

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('figures/eigenval_fidelity_bounds'):
            os.makedirs('figures/eigenval_fidelity_bounds')

        save_as = os.path.join('figures/eigenval_fidelity_bounds', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    # eigval bound plots for different fidelities without noise
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=0/210716_162509',
    #     vqe_label='SPA', xmin=0.0, fid_step=0.1, var_step=0.02, save_as='fidelity_lower_bounds_h2_spa_noiseless.pdf')
    #
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=0/210716_162519',
    #     vqe_label='UpCCGSD', xmin=0.0, fid_step=0.1, var_step=0.02,
    #     save_as='fidelity_lower_bounds_h2_upccgsd_noiseless.pdf')

    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=0/210716_162453',
    #     vqe_label='UpCCGSD', fid_step=0.25, var_step=0.0002, save_as='eigval_lower_bounds_lih_upccgsd_noiseless.pdf')

    # eigval bound plots for different fidelities with noise
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=1/210716_170002',
    #     vqe_label='SPA', xmin=0.0, fid_step=0.25, var_step=0.05, save_as='fidelity_lower_bounds_h2_spa_noisy.pdf')
    #
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=1/210716_170301',
    #     vqe_label='UpCCGSD', xmin=0.0, fid_step=0.25, var_step=0.1,
    #     save_as='fidelity_lower_bounds_h2_upccgsd_noisy.pdf')

    # # plot_lower_bounds_different_fidelities(
    # #     results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
    # #     vqe_label='SPA', fid_step=0.25, var_step=0.1, save_as='fidelity_eigval_lower_bounds_lih_spa_noiseless.pdf')
    #
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210716_171413',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.25, save_as='fidelity_eigval_lower_bounds_beh2_spa_noiseless.pdf')

    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=1/210716_162531',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.1, save_as='eigval_lower_bounds_lih_spa_noisy.pdf')

    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429',
    #     vqe_label='UpCCGSD', fid_step=0.25, var_step=0.2, save_as='eigval_lower_bounds_lih_upccgsd_noisy.pdf')
    #
    # plot_lower_bounds_different_fidelities(
    #     results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=1/210716_171529',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.2, save_as='eigval_lower_bounds_beh2_spa_noisy.pdf')

    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.1, save_as='fidelity_eigval_lower_bounds_lih_spa_noiseless2.pdf')

    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210716_171413',
    #     vqe_label='SPA', fid_step=0.25,
    #     var_step=0.1, save_as='fidelity_eigval_lower_bounds_beh2_spa_noiseless2.pdf')

    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210716_163429',
    #     vqe_label='UpCCGSD', fid_step=0.25, var_step=0.1)

    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
    #     vqe_label='SPA', fid_step=0.05, var_step=0.1)

    plot_lower_bounds_different_fidelities2(
        results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=1/210716_162531',
        vqe_label='SPA', fid_step=0.05, var_step=0.1)

    #
    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210716_162438',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.1)
    #
    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210716_171413',
    #     vqe_label='SPA', fid_step=0.25,
    #     var_step=0.1)

    # plot_lower_bounds_different_fidelities2(
    #     results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=1/210716_171529',
    #     vqe_label='SPA', fid_step=0.25, var_step=0.2)
