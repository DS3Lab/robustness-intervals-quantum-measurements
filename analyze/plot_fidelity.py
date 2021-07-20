import numpy as np
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns

from lib.robustness_interval import GramianEigenvalueInterval, GramianExpectationBound, SDPInterval

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
mcl_color = colors[0]
ours_color = colors[1]

gap_color = colors[2]
variance_color = colors[3]

lw = 1.0


def main(results_dir, molecule, save_as=None):
    stats_df = pd.read_pickle(os.path.join(results_dir, 'statistics.pkl'))

    plot_df = pd.DataFrame(
        columns=['r', 'variance', 'spectral_gap', 'true_fidelity', 'mcclean_fidelity', 'fidelity_bound2'])

    # compute gramian eigenvalue intervals
    for r in stats_df.index:

        vqe_energy = np.mean(stats_df.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df.loc[r]['variances_hamiltonian'])
        spectral_gap = stats_df.loc[r]['E1'] - stats_df.loc[r]['E0']

        data = [r, vqe_variance, spectral_gap]

        # fidelity approximations
        true_fidelity = stats_df.loc[r]['gs_fidelity']
        if vqe_energy <= 0.5 * (stats_df.loc[r]['E0'] + stats_df.loc[r]['E1']):
            mcclean_bound = max(0.5, 1 - np.sqrt(vqe_variance) / spectral_gap)
            fidelity_bound = 0.5 * (1 + np.sqrt(1.0 - min(1, 4 * vqe_variance / (spectral_gap ** 2))))
        else:
            mcclean_bound = 0
            fidelity_bound = 0

        # add fidelity values
        data += [true_fidelity, mcclean_bound, fidelity_bound]

        plot_df.loc[-1] = data
        plot_df.index += 1

    plot_df.sort_values('r', inplace=True)
    plot_df.set_index('r', inplace=True)

    # select subset
    xticks = np.arange(start=0, stop=max(plot_df.index) + 0.5 / 2.0, step=1.0)
    xticks = xticks[xticks >= plot_df.index.min()]

    # setup plot
    fig = plt.figure(constrained_layout=True, figsize=(7, 6))
    gs = fig.add_gridspec(4, 1)
    gs.update(wspace=0.025, hspace=0.1)

    # plot fidelity
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.plot(plot_df.index, plot_df['true_fidelity'], color='black', lw=lw, label='True Fidelity')
    ax1.plot(plot_df[plot_df['mcclean_fidelity'] > 0.5].index,
             plot_df[plot_df['mcclean_fidelity'] > 0.5]['mcclean_fidelity'], color=mcl_color, lw=lw, label='Eq. (11)')
    ax1.plot(plot_df[plot_df['fidelity_bound2'] > 0.5].index,
             plot_df[plot_df['fidelity_bound2'] > 0.5]['fidelity_bound2'], color=ours_color, lw=lw, label='Eq. (12)')

    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'Fidelity')
    loc = plticker.MultipleLocator(base=0.25)
    ax1.yaxis.set_major_locator(loc)

    ax1.legend(fontsize=12, framealpha=0.75)

    # plot spectral gap
    ax2 = fig.add_subplot(gs[2:, 0])
    ax2.plot(plot_df.index, plot_df['spectral_gap'], color=gap_color, lw=lw, ls='-', label='Spectral Gap')
    ax2.set_xticks(xticks)
    ax2.set_ylabel(r'$E_1 - E_0$')
    ax2.set_xlabel('Bond Distance (A)')
    loc = plticker.MultipleLocator(base=0.1)
    ax2.yaxis.set_major_locator(loc)

    # plot variance
    ax3 = ax2.twinx()
    ax3.plot(plot_df.index, plot_df['variance'], color=variance_color, lw=lw, ls='--', label=r'Variance')
    ax3.set_ylabel(r'Variance')

    # legend
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(handles2 + handles3, labels2 + labels3, loc='center left', ncol=1, fontsize=12, framealpha=0.75)

    # add grid
    ax1.grid()
    ax2.grid()

    fig.align_ylabels([ax1, ax2, ax3])

    fig.suptitle(r'Fidelity Bounds for {}'.format(molecule))

    if save_as is None:
        plt.show()
    else:
        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        save_as = os.path.join('./plots', save_as)
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.1)
        pass


if __name__ == '__main__':
    main('../../results2/lih/basis-set-free/hcb=False/spa/noise=0/210706_185138', molecule='LiH',
         save_as='fidelity_lih_spa.pdf')
    main('../../results2/beh2/basis-set-free/hcb=False/spa/noise=0/210706_193706', molecule='BeH$_2$',
         save_as='fidelity_beh2_spa.pdf')

    # main('../../results2/lih/basis-set-free/hcb=False/upccgsd/noise=0/210706_185204', molecule='LiH')
    # main('../../results2/beh2/basis-set-free/hcb=False/upccgsd/noise=0/210714_161456', molecule='LiH')
