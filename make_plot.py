import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os
import pandas as pd
import seaborn as sns

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

fci_color = colors[4]
mp2_color = colors[5]
ccsd_color = colors[6]

vqe_color = colors[0]
eigen_color = colors[1]
mom1_color = colors[2]
mom2_color = colors[3]

figures_path = '/Users/maurice/Dropbox/Apps/Overleaf/[Quantum] NISQ Performance Guarantees/figures'


def make_plot(results_dir, xmin=0.0, xstep=None, error_step=None, err_min=None, err_max=None, fid_step=None,
              energy_step=None, title: str = None, vqe_label=None, save_dir=None, single_bound=False):
    # load and filter data
    energies_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))
    energies_df = energies_df[energies_df.index >= xmin]

    eigen_intervals_df = pd.read_pickle(os.path.join(results_dir, 'eigen_interval.pkl'))
    eigen_intervals_df = eigen_intervals_df[eigen_intervals_df.index >= xmin]

    first_moment_intervals_df = pd.read_pickle(os.path.join(results_dir, 'interval_first_moment.pkl'))
    first_moment_intervals_df = first_moment_intervals_df[first_moment_intervals_df.index >= xmin]

    second_moment_intervals_df = pd.read_pickle(os.path.join(results_dir, 'interval_second_moment.pkl'))
    second_moment_intervals_df = second_moment_intervals_df[second_moment_intervals_df.index >= xmin]

    dists = energies_df.index

    xticks = np.arange(start=0, stop=max(dists) + xstep / 2.0, step=xstep)
    xticks = xticks[xticks >= min(dists)]

    # setup figure
    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(6, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energies and bounds
    ax1 = fig.add_subplot(gs[:4, 0])
    ax1.plot(dists, energies_df['exact'], label='exact', color='black')
    ax1.plot(dists, energies_df['vqe'], label=vqe_label or 'vqe', color=vqe_color, marker='o', markersize=3,
             linewidth=0.75)

    # eigenvalue bound
    ax1.plot(dists, eigen_intervals_df['lower_ci'], label='Eq. (14)', linestyle='-', color=eigen_color, marker='x',
             markersize=3, linewidth=0.75)

    if not single_bound:
        # first moment bound
        ax1.plot(dists, first_moment_intervals_df['lower_mean'], label='Expec1', linestyle='-', color=mom1_color,
                 marker='x', markersize=3, linewidth=0.75)

        # second moment bound
        ax1.plot(dists, second_moment_intervals_df['lower_ci'], label='Expec2', linestyle='-', color=mom2_color,
                 marker='x',
                 markersize=3, linewidth=0.75)

    # format axes
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Energy')
    loc = plticker.MultipleLocator(base=energy_step or 0.1)
    ax1.yaxis.set_major_locator(loc)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # plot error vqe - E_0 and E_0 - lower_bound
    vqe_errors = energies_df['vqe'] - energies_df['exact']
    bound0_errors = energies_df['exact'] - eigen_intervals_df['lower_ci']
    bound1_errors = energies_df['exact'] - first_moment_intervals_df['lower_mean']
    bound2_errors = energies_df['exact'] - second_moment_intervals_df['lower_ci']

    ax2 = fig.add_subplot(gs[4, 0])
    ax2.plot(dists, vqe_errors, label=r'vqe - E$_0$', color=vqe_color, linestyle='', marker='x', markersize=3,
             linewidth=0.75)
    ax2.plot(dists, bound0_errors, label=r'E$_0$ - Eq.(14)', color=eigen_color, linestyle='', marker='x', markersize=3,
             linewidth=0.75)

    if not single_bound:
        ax2.plot(dists, bound1_errors, label=r'E$_0$ - Expec1', color=mom1_color, linestyle='', marker='x',
                 markersize=3,
                 linewidth=0.75)
        ax2.plot(dists, bound2_errors, label=r'E$_0$ - Expec2', color=mom2_color, linestyle='', marker='x',
                 markersize=3,
                 linewidth=0.75)

    # chemical accuracy
    if single_bound:
        ax2.axhspan(0.0015, 0.00, color='gray', alpha=0.5)

    # format axes
    ax2.set_xticks(xticks)
    ax2.xaxis.set_ticklabels([])
    loc = plticker.MultipleLocator(base=error_step or 0.05)
    ax2.yaxis.set_major_locator(loc)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel(r'Error')
    ax2.set_ylim(ymin=err_min, ymax=err_max)

    # plot fidelity
    ax3 = fig.add_subplot(gs[5, 0])
    ax3.plot(dists, energies_df['gs_fidelity'], color=vqe_color, linestyle='-', marker='o', markersize=3,
             linewidth=0.75)

    # format axes
    ax3.set_xticks(xticks)
    ax3.set_ylabel(r'Fidelity')
    loc = plticker.MultipleLocator(base=fid_step or 0.1)
    ax3.yaxis.set_major_locator(loc)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_ylim(ymax=1 + 0.02 * (energies_df['gs_fidelity'].max() - energies_df['gs_fidelity'].min()))

    # add grid
    ax1.grid()
    ax2.grid()
    ax3.grid()

    # title
    if title is not None:
        fig.suptitle(title)

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=2, fontsize=14, framealpha=0.75)

    axes = [ax1, ax2, ax3]
    fig.align_ylabels(axes)

    if save_dir is not None:
        if single_bound:
            title = 'eigenvalue_bound_' + title

        save_as = os.path.join(save_dir, title.replace('$', '').replace('+', '_').replace(' ', '') + '.pdf')
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)

        return

    plt.show()


def make_plot_v2(results_dir, xmin=0.0, xstep=None, error_step=None, err_min=None, err_max=None, fid_step=None,
                 energy_step=None, title: str = None, vqe_label=None, save_dir=None):
    # load and filter data
    energies_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))
    energies_df = energies_df[energies_df.index >= xmin]

    eigen_intervals_df = pd.read_pickle(os.path.join(results_dir, 'eigen_interval.pkl'))
    eigen_intervals_df = eigen_intervals_df[eigen_intervals_df.index >= xmin]

    dists = energies_df.index

    xticks = np.arange(start=0, stop=max(dists) + xstep / 2.0, step=xstep)
    xticks = xticks[xticks >= min(dists)]

    # setup figure
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energies and bounds
    ax1 = fig.add_subplot(gs[:4, 0])
    ax1.plot(dists, energies_df['exact'], label='exact', color='black')
    ax1.plot(dists, energies_df['vqe'], label=vqe_label or 'vqe', color=vqe_color, marker='o', markersize=3,
             linewidth=0.75)

    # eigenvalue bound
    ax1.fill_between(dists, eigen_intervals_df['lower_ci'], eigen_intervals_df['upper_ci'],
                     label='Robustness Interval Eq. (14)', linestyle='-', color=eigen_color, alpha=0.25, linewidth=0.75)
    ax1.plot(dists, eigen_intervals_df['lower_ci'], label=None, color=eigen_color, marker='x', markersize=3,
             linewidth=0.75)
    ax1.plot(dists, eigen_intervals_df['upper_ci'], label=None, color=eigen_color, marker='x', markersize=3,
             linewidth=0.75)

    # format axes
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Energy')
    loc = plticker.MultipleLocator(base=energy_step or 0.1)
    ax1.yaxis.set_major_locator(loc)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # plot fidelity
    ax2 = fig.add_subplot(gs[4, 0])
    ax2.plot(dists, energies_df['gs_fidelity'], color=vqe_color, linestyle='-', marker='o', markersize=3,
             linewidth=0.75)

    # format axes
    ax2.set_xticks(xticks)
    ax2.set_ylabel(r'Fidelity')
    loc = plticker.MultipleLocator(base=fid_step or 0.1)
    ax2.yaxis.set_major_locator(loc)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # add grid
    ax1.grid()
    ax2.grid()

    # title
    if title is not None:
        fig.suptitle(title)

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=2, fontsize=14, framealpha=0.75)

    fig.align_ylabels([ax1, ax2])

    if save_dir is not None:
        title = title if title is not None else 'robustness_interval_lih'
        save_as = os.path.join(save_dir, title.replace('$', '').replace('+', '_').replace(' ', '') + '.pdf')
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
        return

    plt.show()


def make_bound_comparison_plots(save):
    noisy_configs = {'h2': {
        'upccgsd': dict(results_dir='../results/h2/basis-set-free/hcb=False/upccgsd/noise=1/210705_153838',
                        xmin=0.0, xstep=0.25, error_step=0.1, fid_step=0.1, energy_step=0.1,
                        title=r'H$_2$ + UpCCGSD + noise', vqe_label='UpCCGSD', err_min=-0.01),
        'spa': dict(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=1/210704_171849',
                    xmin=0.0, xstep=0.25, error_step=0.05, fid_step=0.025, energy_step=0.1,
                    title=r'H$_2$ + SPA + noise', vqe_label='SPA', err_min=-0.01)},
        'lih': {
            'upccgsd': dict(results_dir='../results/lih/basis-set-free/hcb=False/upccgsd/noise=1/210705_160508',
                            xmin=0.0, xstep=0.5, error_step=0.1, fid_step=0.1, energy_step=0.2,
                            title='LiH + UpCCGSD + noise', vqe_label='UpCCGSD', err_min=-0.01),
            'spa': dict(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=1/210704_171927',
                        xmin=0.0, xstep=0.5, error_step=0.1, fid_step=0.1, energy_step=0.2,
                        title='LiH + SPA + noise', vqe_label='SPA', err_min=-0.01)},
        'beh2': {
            'spa': dict(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=1/210704_172032',
                        xmin=0.0, xstep=0.5, error_step=0.5, fid_step=0.25, energy_step=0.5,
                        title=r'BeH$_2$ + SPA + noise', vqe_label='SPA', err_min=-0.01)}}

    noiseless_configs = {'h2': {
        'spa': dict(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=0/210704_171642',
                    xmin=0.0, xstep=0.25, error_step=0.00025, fid_step=0.00000025, energy_step=0.05,
                    title=r'H$_2$ + SPA', vqe_label='SPA')},
        'lih': {
            'spa': dict(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210704_171658',
                        xmin=0.9, xstep=0.5, error_step=0.1, fid_step=0.025, energy_step=0.05,
                        title='LiH + SPA', vqe_label='SPA')},
        'beh2': {
            'spa': dict(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210704_171718',
                        xmin=0.0, xstep=0.5, error_step=0.5, fid_step=0.25, energy_step=0.5,
                        title=r'BeH$_2$ + SPA', vqe_label='SPA')}}

    save_dir = None if not save else figures_path

    # upccgsd plots
    make_plot(**noisy_configs['h2']['upccgsd'], save_dir=save_dir)
    make_plot(**noisy_configs['lih']['upccgsd'], save_dir=save_dir)

    # spa plots
    make_plot(**noisy_configs['h2']['spa'], save_dir=save_dir)
    make_plot(**noisy_configs['lih']['spa'], save_dir=save_dir)
    make_plot(**noisy_configs['beh2']['spa'], save_dir=save_dir)
    make_plot(**noiseless_configs['h2']['spa'], save_dir=save_dir)
    make_plot(**noiseless_configs['lih']['spa'], save_dir=save_dir)
    make_plot(**noiseless_configs['beh2']['spa'], save_dir=save_dir)


def make_single_bound_plot(save):
    noisy_configs = {'h2': {
        'spa': dict(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=1/210704_171849',
                    xmin=0.0, xstep=0.25, error_step=0.025, fid_step=0.025, energy_step=0.05,
                    title=r'H$_2$ + SPA + noise', vqe_label='SPA', err_min=-0.00005)},
        'lih': {
            'spa': dict(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=1/210704_171927',
                        xmin=0.9, xstep=0.5, error_step=0.025, fid_step=0.1, energy_step=0.05,
                        title='LiH + SPA + noise', vqe_label='SPA', err_min=-0.001)},
        'beh2': {
            'spa': dict(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=1/210704_172032',
                        xmin=0.0, xstep=0.5, error_step=0.1, fid_step=0.25, energy_step=0.1,
                        title=r'BeH$_2$ + SPA + noise', vqe_label='SPA', err_min=-0.01)}}

    noiseless_configs = {'h2': {
        'spa': dict(results_dir='../results/h2/basis-set-free/hcb=False/spa/noise=0/210704_171642',
                    xmin=0.0, xstep=0.25, error_step=0.001, fid_step=0.00000025, energy_step=0.05,
                    title=r'H$_2$ + SPA', vqe_label='SPA', err_min=-0.0001)},
        'lih': {
            'spa': dict(results_dir='../results/lih/basis-set-free/hcb=False/spa/noise=0/210704_171658',
                        xmin=0.9, xstep=0.5, error_step=0.005, fid_step=0.025, energy_step=0.025,
                        title='LiH + SPA', vqe_label='SPA', err_min=-0.001)},
        'beh2': {
            'spa': dict(results_dir='../results/beh2/basis-set-free/hcb=False/spa/noise=0/210704_171718',
                        xmin=0.0, xstep=0.5, error_step=0.01, fid_step=0.25, energy_step=0.1,
                        title=r'BeH$_2$ + SPA', vqe_label='SPA', err_max=0.025, err_min=-0.001)}}

    save_dir = None if not save else figures_path

    # spa plots
    make_plot(**noisy_configs['h2']['spa'], save_dir=save_dir, single_bound=True)
    make_plot(**noisy_configs['lih']['spa'], save_dir=save_dir, single_bound=True)
    make_plot(**noisy_configs['beh2']['spa'], save_dir=save_dir, single_bound=True)
    make_plot(**noiseless_configs['h2']['spa'], save_dir=save_dir, single_bound=True)
    make_plot(**noiseless_configs['lih']['spa'], save_dir=save_dir, single_bound=True)
    make_plot(**noiseless_configs['beh2']['spa'], save_dir=save_dir, single_bound=True)


if __name__ == '__main__':
    # make_bound_comparison_plots(True)
    make_single_bound_plot(False)

    # make_plot(**noisy_configs['h2']['spa'], save_dir=figures_path)
    # make_plot(**noisy_configs['h2']['upccgsd'], save_dir=figures_path)
    # make_plot(**noisy_configs['lih']['spa'], save_dir=figures_path)
    # make_plot(**noisy_configs['lih']['upccgsd'], save_dir=figures_path)
    # make_plot(**noisy_configs['beh2']['spa'], save_dir=figures_path)

    # make_plot(**noiseless_configs['h2']['spa'], save_dir=figures_path)
    # make_plot(**noiseless_configs['lih']['spa'], save_dir=None)
    # make_plot(**noiseless_configs['beh2']['spa'], save_dir=figures_path)

    # kwargs = noisy_configs['lih']['upccgsd']
    # kwargs.update(fid_step=0.05)
    # kwargs.update(title=None)
    # make_plot_v2(**kwargs, save_dir=figures_path)

# def load_data_csv(csv_fn):
#     return pd.read_csv(csv_fn).sort_values(by='r', ascending=True)
#
#
# def load_data_pickle(pkl_fn):
#     df = pd.read_pickle(pkl_fn)
#     return df
#
#     # return {'dists': df.index.to_list(),
#     #         'exact': df.get('exact').to_list(),
#     #         'vqe': df.get('vqe_mean').to_list(),
#     #         'fidelity': df.get('gs_fidelity').to_list(),
#     #         'lower_mean': df.get('lower_mean').to_list(),
#     #         'upper_mean': df.get('upper_mean').to_list(),
#     #         'lower_mean_ci': df.get('lower_ci').to_list(),
#     #         'upper_mean_ci': df.get('upper_ci').to_list()}

# def make_plot_OLD(pkl_fn, xstep, xmin, ymin1=None, ymax1=None, ystep2=None, ymin2=None, ymax2=None):
#     data = load_data_pickle(pkl_fn)
#     data = data[data.index >= xmin]
#     dists = data.index.to_list()
#
#     fig = plt.figure(constrained_layout=True, figsize=(7, 5))
#     gs = fig.add_gridspec(5, 1)
#     gs.update(wspace=0.025, hspace=0.1)
#
#     # plot energies
#     ax1 = fig.add_subplot(gs[:4, 0])
#     ax1.plot(dists, data['exact'], label='exact', color='black')
#     # ax1.plot(dists, data_spa['fci'], label=r'fci', linestyle='-', color=fci_color)
#     # ax1.plot(dists, data_spa['mp2'], label=r'mp2', linestyle='-', zorder=1, color=mp2_color)
#     # ax1.plot(dists, data_spa['ccsd'], label='ccsd', linestyle='-', zorder=1, color=ccsd_color)
#     # ax1.plot(dists, data_spa['vqe'], label='SPA', linestyle='-', marker='o', zorder=2, color=spa_color, markersize=4)
#     ax1.plot(dists, data['vqe_mean'], label='SPA', linestyle='-', marker='x', zorder=2, color=spas_color,
#              markersize=4)
#
#     # plot bounds
#     # ax1.plot(dists, data['lower_mean'], label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
#     #          color=spa_color, marker='o', markersize=4)
#     lower_ci = np.array(data['lower_ci'].to_list())[:, 0]
#     ax1.plot(dists, lower_ci, label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
#              color=spa_color, marker='o', markersize=4)
#     # ax1.plot(dists, data['lower_mean'], label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
#     #          color=spa_color, marker='o', markersize=4)
#     # ax1.plot(dists, data_spas['eig_lower'], label='SPA+S Lower Bound', alpha=0.75, linestyle='--', zorder=2,
#     #          color=spas_color, marker='x', markersize=4)
#
#     # xticks
#     xticks = np.arange(start=0, stop=max(dists) + xstep / 2.0, step=xstep)
#     xticks = xticks[xticks >= min(dists)]
#     ax1.set_xticks(xticks)
#     ax1.xaxis.set_ticklabels([])
#     ax1.set_ylim((ymin1, ymax1))
#     ax1.set_ylabel('Energy')
#
#     # plot fidelity values
#     ax2 = fig.add_subplot(gs[4, 0])
#     ax2.plot(dists, data['gs_fidelity'], linestyle='-', marker='o', markersize=4, zorder=2, color=spa_color)
#     # ax2.plot(dists, data_spas['fidelity'], linestyle='-', marker='x', markersize=4, zorder=2, color=spas_color)
#
#     # xticks
#     ax2.set_xticks(xticks)
#     ax2.xaxis.set_ticklabels(xticks)
#
#     # yticks
#     yticks = np.arange(0, 1 + (ystep2 or 0.2) / 2, step=ystep2 or 0.2)
#     yticks = yticks[yticks >= ymin2]
#     ax2.set_yticks(yticks)
#     ax2.set_ylim((ymin2, ymax2))
#     ax2.set_xticks(xticks)
#     ax2.set_ylabel('Fidelity')
#
#     # format
#     handles, labels = ax1.get_legend_handles_labels()
#     ax1.legend(handles, labels, loc='best', ncol=1, fontsize=14, framealpha=0.75)
#
#     ax1.grid()
#     ax2.grid()
#
#     plt.show()


# def make_interval_plot(csv_spa, xstep, xmin, ymin1=None, ymax1=None, ystep2=None, ymin2=None, ymax2=None):
#     # load and filter data
#     data_spa = load_data(csv_spa)
#     data_spa = data_spa[data_spa['r'] >= xmin]
#
#     dists = data_spa['r']
#
#     fig = plt.figure(constrained_layout=True, figsize=(7, 5))
#     gs = fig.add_gridspec(5, 1)
#     gs.update(wspace=0.025, hspace=0.1)
#
#     # plot energies
#     ax1 = fig.add_subplot(gs[:4, 0])
#     ax1.plot(dists, data_spa['exact'], label='exact', color='black')
#     ax1.plot(dists, data_spa['vqe'], label='SPA', linestyle='-', marker='o', zorder=2, color=spa_color, markersize=4)
#
#     # plot bounds
#     ax1.fill_between(dists, data_spa['eig_lower'], data_spa['eig_upper'], alpha=0.25, color=spa_color)
#
#     # xticks
#     xticks = np.arange(start=0, stop=max(dists) + xstep / 2.0, step=xstep)
#     xticks = xticks[xticks >= min(dists)]
#     ax1.set_xticks(xticks)
#     ax1.xaxis.set_ticklabels([])
#     ax1.set_ylim((ymin1, ymax1))
#     ax1.set_ylabel('Energy')
#
#     # plot fidelity values
#     ax2 = fig.add_subplot(gs[4, 0])
#     ax2.plot(dists, data_spa['fidelity'], linestyle='-', marker='o', markersize=4, zorder=2, color=spa_color)
#
#     # xticks
#     ax2.set_xticks(xticks)
#     ax2.xaxis.set_ticklabels(xticks)
#
#     # yticks
#     yticks = np.arange(0, 1 + (ystep2 or 0.2) / 2, step=ystep2 or 0.2)
#     yticks = yticks[yticks >= ymin2]
#     ax2.set_yticks(yticks)
#     ax2.set_ylim((ymin2, ymax2))
#     ax2.set_xticks(xticks)
#     ax2.set_ylabel('Fidelity')
#
#     # format
#     handles, labels = ax1.get_legend_handles_labels()
#     ax1.legend(handles, labels, loc='upper left', ncol=1, fontsize=14, framealpha=0.5)
#
#     ax1.grid()
#     ax2.grid()
#
#     plt.show()


# if __name__ == '__main__':
#     # csv_spa = 'results/beh2/basis-set-free/hcb=False/spa/noise=0/use_grouping=1/normalize_hamiltonian=0/results.csv'
#     # csv_spas = 'results/beh2/basis-set-free/hcb=False/spa-s/noise=0/use_grouping=1/normalize_hamiltonian=0/results.csv'
#     # # make_plot(csv_spa, csv_spas, xstep=0.5, xmin=0.75, ystep2=0.3, ymin2=0.1, ymax2=1.05)
#     # make_interval_plot(csv_spa, xstep=0.5, xmin=0.0, ystep2=0.3, ymin2=0.1, ymax2=1.05)
#     pkl_file = 'results/lih/basis-set-free/hcb=False/spa/noise=1/210703_113741/interval_second_moment.pkl'
#     make_plot(pkl_file, xstep=0.5, xmin=0.75, ymin2=0.8, ystep2=0.1)
