import matplotlib as mpl
import seaborn as sns


def init_plot_style():
    sns.set_style('ticks')
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = FONT_SIZE_LARGE
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["axes.linewidth"] = 1.
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.


FONT_SIZE = 12
FONT_SIZE_LARGE = 16
LINEWIDTH = 1.0
FIGSIZE = (6, 4.32)
COLORS = sns.color_palette('muted')

VQE_COLOR = COLORS[0]
GRAMIAN_EIGVAL_COLOR = COLORS[1]
GRAMIAN_EXPECTATION_COLOR = COLORS[2]
SDP_COLOR = COLORS[3]

GRAM_EIGVAL_KWARGS = dict(linestyle='--', marker='o', markersize=3, markerfacecolor='none', lw=LINEWIDTH,
                          color=GRAMIAN_EIGVAL_COLOR)
GRAM_EXPEC_KWARGS = dict(linestyle='--', marker='o', markersize=3, markerfacecolor='none', lw=LINEWIDTH,
                         color=GRAMIAN_EXPECTATION_COLOR)
SDP_KWARGS = dict(linestyle='--', marker='o', markersize=3, markerfacecolor='none', lw=LINEWIDTH, color=SDP_COLOR)
VQE_LINE_KWARGS = dict(linestyle='-', marker='o', markersize=3, markerfacecolor='none', lw=LINEWIDTH, color=VQE_COLOR)
