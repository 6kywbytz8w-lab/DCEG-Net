import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

import matplotlib.dates as mdates

CEEMDAN_PLOT_DIR = os.path.join(SAVE_DIR, 'ceemdan_pretty_plots')

os.makedirs(CEEMDAN_PLOT_DIR, exist_ok=True)

CITY_NAME_EN = {'长春': 'Changchun', '吉林': 'Jilin', '四平': 'Siping', '辽源': 'Liaoyuan', '通化': 'Tonghua', '白山': 'Baishan', '松原': 'Songyuan', '白城': 'Baicheng', '延边': 'Yanbian'}

CEEMDAN_COLORS = {'signal': '#E79BB2', 'residue': '#C7A6F5', 'grid': '#EAF7F1', 'spine': '#D7EEE5', 'text': '#4F6475'}

IMF_PASTEL_COLORS = ['#7ECBEF', '#8FD8B8', '#F6C38B', '#A9B8F8', '#9EDDD6', '#F4B5A6', '#B8E39A', '#DAB6F8']

plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 320, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'axes.unicode_minus': False, 'axes.labelsize': 13, 'axes.titlesize': 15, 'xtick.labelsize': 10.0, 'ytick.labelsize': 10.0, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def safe_filename(name: str) -> str:
    invalid_chars = '\\\\/:*?"<>|'
    for ch in invalid_chars:
        name = name.replace(ch, '_')
    return str(name)

def _style_ceemdan_axis(ax, show_x=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CEEMDAN_COLORS['spine'])
    ax.spines['bottom'].set_color(CEEMDAN_COLORS['spine'])
    ax.grid(axis='y', linestyle='--', linewidth=0.7, color=CEEMDAN_COLORS['grid'], alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', which='both', labelleft=False, left=False)
    if show_x:
        locator = mdates.MonthLocator(interval=4)
        formatter = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', which='major', labelbottom=True, bottom=True, labelsize=10, colors=CEEMDAN_COLORS['text'], rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    else:
        ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

def _draw_pretty_line(ax, x, y, color, label_text, lw=1.5, fill_alpha=0.1):
    line, = ax.plot(x, y, color=color, linewidth=lw, zorder=3, solid_capstyle='round', solid_joinstyle='round')
    line.set_path_effects([pe.Stroke(linewidth=lw + 0.8, foreground='white', alpha=0.92), pe.Normal()])
    ax.fill_between(x, 0, y, color=color, alpha=fill_alpha, zorder=1)
    ax.axhline(0, color='#F7FBF8', linewidth=0.8, zorder=0)
    ax.text(0.012, 0.82, label_text, transform=ax.transAxes, ha='left', va='center', fontsize=10.2, color=CEEMDAN_COLORS['text'], bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=color, linewidth=0.9, alpha=0.92))

def plot_city_ceemdan_from_current_script(city_name, city_idx):
    city_name_en = CITY_NAME_EN.get(city_name, city_name)
    signal = pivot_aqi.loc[dates, city_name].values.astype(np.float32)
    comps = np.asarray(city_component_list[city_idx], dtype=np.float32)
    imf_counts = np.asarray(city_imf_count_list[city_idx], dtype=int)
    valid_counts = imf_counts[imf_counts >= 0]
    if valid_counts.size == 0:
        raise ValueError(f'No complete rolling CEEMDAN windows for {city_name_en}.')
    n_imfs = int(valid_counts.max())
    n_rows = 1 + n_imfs + 1
    fig_h = 1.45 * n_rows + 0.9
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.6, fig_h), sharex=True, gridspec_kw={'hspace': 0.14})
    _draw_pretty_line(axes[0], dates, signal, color=CEEMDAN_COLORS['signal'], label_text='Original AQI', lw=1.8, fill_alpha=0.1)
    _style_ceemdan_axis(axes[0], show_x=False)
    valid_dates = imf_counts >= 0
    for k in range(n_imfs):
        line_values = comps[k].astype(float).copy()
        line_values[(imf_counts <= k) | ~valid_dates] = np.nan
        comp_color = IMF_PASTEL_COLORS[k % len(IMF_PASTEL_COLORS)]
        _draw_pretty_line(axes[k + 1], dates, line_values, color=comp_color, label_text=f'IMF {k + 1}', lw=1.35, fill_alpha=0.09)
        _style_ceemdan_axis(axes[k + 1], show_x=False)
    residue_values = comps[-1].astype(float).copy()
    residue_values[~valid_dates] = np.nan
    _draw_pretty_line(axes[-1], dates, residue_values, color=CEEMDAN_COLORS['residue'], label_text='Residue', lw=1.6, fill_alpha=0.1)
    _style_ceemdan_axis(axes[-1], show_x=True)
    fig.suptitle(f'{city_name_en} - Causal Rolling CEEMDAN Endpoint Coefficients', fontsize=14, color=CEEMDAN_COLORS['text'], y=0.995)
    axes[-1].set_xlabel('Date', color=CEEMDAN_COLORS['text'])
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    file_base = f'ceemdan_{safe_filename(city_name_en)}'
    fig.savefig(os.path.join(CEEMDAN_PLOT_DIR, f'{file_base}.png'), bbox_inches='tight', dpi=320)
    fig.savefig(os.path.join(CEEMDAN_PLOT_DIR, f'{file_base}.pdf'), bbox_inches='tight')
    plt.close(fig)

for city_idx, city in enumerate(city_order):
    plot_city_ceemdan_from_current_script(city, city_idx)

print('各市 CEEMDAN 美化图已保存到：', CEEMDAN_PLOT_DIR)
