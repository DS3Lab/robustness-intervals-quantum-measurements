import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

font_size = 14
font_size_large = 20
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

fci_color = colors[3]
mp2_color = colors[4]
ccsd_color = colors[5]

spa_color = colors[0]
spas_color = colors[1]
upccgsd_color = colors[2]


def load_data_csv(csv_fn):
    return pd.read_csv(csv_fn).sort_values(by='r', ascending=True)


def load_data_pickle(pkl_fn):
    df = pd.read_pickle(pkl_fn)
    return df

    # return {'dists': df.index.to_list(),
    #         'exact': df.get('exact').to_list(),
    #         'vqe': df.get('vqe_mean').to_list(),
    #         'fidelity': df.get('gs_fidelity').to_list(),
    #         'lower_mean': df.get('lower_mean').to_list(),
    #         'upper_mean': df.get('upper_mean').to_list(),
    #         'lower_mean_ci': df.get('lower_ci').to_list(),
    #         'upper_mean_ci': df.get('upper_ci').to_list()}


def make_plot(pkl_fn, xstep, xmin, ymin1=None, ymax1=None, ystep2=None, ymin2=None, ymax2=None):
    # # load and filter data
    # data_spa = load_data(csv_spa)
    # data_spa = data_spa[data_spa['r'] >= xmin]
    #
    # data_spas = load_data(csv_spas)
    # data_spas = data_spas[data_spas['r'] >= xmin]
    #
    # dists = data_spa['r']

    data = load_data_pickle(pkl_fn)
    data = data[data.index >= xmin]
    dists = data.index.to_list()

    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot energies
    ax1 = fig.add_subplot(gs[:4, 0])
    ax1.plot(dists, data['exact'], label='exact', color='black')
    # ax1.plot(dists, data_spa['fci'], label=r'fci', linestyle='-', color=fci_color)
    # ax1.plot(dists, data_spa['mp2'], label=r'mp2', linestyle='-', zorder=1, color=mp2_color)
    # ax1.plot(dists, data_spa['ccsd'], label='ccsd', linestyle='-', zorder=1, color=ccsd_color)
    # ax1.plot(dists, data_spa['vqe'], label='SPA', linestyle='-', marker='o', zorder=2, color=spa_color, markersize=4)
    ax1.plot(dists, data['vqe_mean'], label='SPA', linestyle='-', marker='x', zorder=2, color=spas_color,
             markersize=4)

    # plot bounds
    # ax1.plot(dists, data['lower_mean'], label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
    #          color=spa_color, marker='o', markersize=4)
    lower_ci = np.array(data['lower_ci'].to_list())[:,0]
    ax1.plot(dists, lower_ci, label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
             color=spa_color, marker='o', markersize=4)
    # ax1.plot(dists, data['lower_mean'], label='SPA Lower Bound', alpha=0.75, linestyle='--', zorder=2,
    #          color=spa_color, marker='o', markersize=4)
    # ax1.plot(dists, data_spas['eig_lower'], label='SPA+S Lower Bound', alpha=0.75, linestyle='--', zorder=2,
    #          color=spas_color, marker='x', markersize=4)

    # xticks
    xticks = np.arange(start=0, stop=max(dists) + xstep / 2.0, step=xstep)
    xticks = xticks[xticks >= min(dists)]
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylim((ymin1, ymax1))
    ax1.set_ylabel('Energy')

    # plot fidelity values
    ax2 = fig.add_subplot(gs[4, 0])
    ax2.plot(dists, data['gs_fidelity'], linestyle='-', marker='o', markersize=4, zorder=2, color=spa_color)
    # ax2.plot(dists, data_spas['fidelity'], linestyle='-', marker='x', markersize=4, zorder=2, color=spas_color)

    # xticks
    ax2.set_xticks(xticks)
    ax2.xaxis.set_ticklabels(xticks)

    # yticks
    yticks = np.arange(0, 1 + (ystep2 or 0.2) / 2, step=ystep2 or 0.2)
    yticks = yticks[yticks >= ymin2]
    ax2.set_yticks(yticks)
    ax2.set_ylim((ymin2, ymax2))
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Fidelity')

    # format
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best', ncol=1, fontsize=14, framealpha=0.75)

    ax1.grid()
    ax2.grid()

    plt.show()


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


if __name__ == '__main__':
    # csv_spa = 'results/beh2/basis-set-free/hcb=False/spa/noise=0/use_grouping=1/normalize_hamiltonian=0/results.csv'
    # csv_spas = 'results/beh2/basis-set-free/hcb=False/spa-s/noise=0/use_grouping=1/normalize_hamiltonian=0/results.csv'
    # # make_plot(csv_spa, csv_spas, xstep=0.5, xmin=0.75, ystep2=0.3, ymin2=0.1, ymax2=1.05)
    # make_interval_plot(csv_spa, xstep=0.5, xmin=0.0, ystep2=0.3, ymin2=0.1, ymax2=1.05)
    pkl_file = 'results/lih/basis-set-free/hcb=False/spa/noise=1/210703_113741/interval_second_moment.pkl'
    make_plot(pkl_file, xstep=0.5, xmin=0.75, ymin2=0.8, ystep2=0.1)
