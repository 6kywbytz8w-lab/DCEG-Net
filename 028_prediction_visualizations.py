import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.patheffects as pe

TIME_TICK_INTERVAL = 3

SET_NAME_MAP = {'Validation': 'Set1', 'Test': 'Set2'}

PALETTE = {'observed': '#E79BB2', 'dual': '#7ECBEF', 'fusion': '#8FD8B8', 'persistence': '#F6C38B', 'grid': '#EAF7F1', 'spine': '#D7EEE5', 'text': '#4F6475', 'fill_dual': '#DFF4FD'}

plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 320, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'axes.unicode_minus': False, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 11.5, 'axes.linewidth': 1.1, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def save_fig(fig, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{name}.png'), bbox_inches='tight', dpi=320)
    fig.savefig(os.path.join(save_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.show()

def beautify_axis(ax, dates_phase):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.grid(axis='y', linestyle='--', linewidth=0.9, color=PALETTE['grid'], alpha=0.9)
    ax.set_axisbelow(True)
    locator = mdates.MonthLocator(interval=TIME_TICK_INTERVAL)
    formatter = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(min(dates_phase), max(dates_phase))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=PALETTE['text'])
    ax.tick_params(axis='y', which='both', labelleft=False, left=False)
    ax.tick_params(axis='x', which='major', labelbottom=True, bottom=True, colors=PALETTE['text'], length=3.5, width=0.8)

def add_soft_line(ax, x, y, color, lw=2.2, ls='-', alpha=1.0, z=3, label=None):
    line, = ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=z, label=label, solid_capstyle='round', solid_joinstyle='round')
    line.set_path_effects([pe.Stroke(linewidth=lw + 1.6, foreground='white', alpha=0.95), pe.Normal()])
    return line

def plot_province_comparison_fresh(phase_name, y_true, dates_phase, key, save_dir):
    y_obs = province_series(y_true)
    y_dual = province_series(model_dict_eval['DualBranch-CEGNetStyle'][key])
    y_fusion = province_series(model_dict_eval['Fusion'][key])
    y_persist = province_series(model_dict_eval['Persistence'][key])
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    add_soft_line(ax, dates_phase, y_persist, color=PALETTE['persistence'], lw=1.8, ls='--', alpha=0.95, z=2, label='Persistence')
    add_soft_line(ax, dates_phase, y_fusion, color=PALETTE['fusion'], lw=2.0, ls='-.', alpha=0.98, z=3, label='Fusion')
    add_soft_line(ax, dates_phase, y_obs, color=PALETTE['observed'], lw=2.25, ls='-', alpha=0.98, z=4, label='Observed')
    add_soft_line(ax, dates_phase, y_dual, color=PALETTE['dual'], lw=2.9, ls='-', alpha=1.0, z=5, label='DualBranch-CEGNetStyle')
    ax.fill_between(dates_phase, y_obs, y_dual, color=PALETTE['fill_dual'], alpha=0.1, zorder=1)
    beautify_axis(ax, dates_phase)
    ax.set_ylabel('AQI', color=PALETTE['text'])
    ax.set_xlabel('Date', color=PALETTE['text'])
    display_phase = SET_NAME_MAP.get(phase_name, phase_name)
    ax.set_title(f'Province-level AQI Prediction on {display_phase}', color=PALETTE['text'], pad=14)
    handles, labels = ax.get_legend_handles_labels()
    order_map = {'Observed': 0, 'DualBranch-CEGNetStyle': 1, 'Fusion': 2, 'Persistence': 3}
    order = np.argsort([order_map[l] for l in labels])
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False, handlelength=2.8, columnspacing=1.5)
    for txt in leg.get_texts():
        txt.set_color(PALETTE['text'])
    fig.tight_layout()
    save_fig(fig, save_dir, f'province_comparison_{phase_name.lower()}_fresh')

def plot_dualbranch_fit_fresh(phase_name, y_true, y_pred, dates_phase, save_dir):
    y_obs = province_series(y_true)
    y_hat = province_series(y_pred)
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.fill_between(dates_phase, y_obs, y_hat, color=PALETTE['fill_dual'], alpha=0.18, zorder=1)
    add_soft_line(ax, dates_phase, y_obs, color=PALETTE['observed'], lw=2.35, ls='-', alpha=0.98, z=4, label='Observed')
    add_soft_line(ax, dates_phase, y_hat, color=PALETTE['dual'], lw=3.0, ls='-', alpha=1.0, z=5, label='Predicted (DualBranch-CEGNetStyle)')
    beautify_axis(ax, dates_phase)
    ax.set_ylabel('AQI', color=PALETTE['text'])
    ax.set_xlabel('Date', color=PALETTE['text'])
    display_phase = SET_NAME_MAP.get(phase_name, phase_name)
    ax.set_title(f'DualBranch-CEGNetStyle Fit Against Observations on {display_phase}', color=PALETTE['text'], pad=14)
    leg = ax.legend(loc='upper right', frameon=False, handlelength=2.8)
    for txt in leg.get_texts():
        txt.set_color(PALETTE['text'])
    fig.tight_layout()
    save_fig(fig, save_dir, f'dualbranch_province_fit_{phase_name.lower()}_fresh')

def plot_dualbranch_scatter_fresh(phase_name, y_true, y_pred, save_dir):
    y_obs = province_series(y_true).reshape(-1)
    y_hat = province_series(y_pred).reshape(-1)
    low = min(np.min(y_obs), np.min(y_hat))
    high = max(np.max(y_obs), np.max(y_hat))
    pad = (high - low) * 0.05
    low -= pad
    high += pad
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_obs, y_hat, s=42, color=PALETTE['dual'], alpha=0.78, edgecolors='white', linewidths=0.7, zorder=3)
    ax.plot([low, high], [low, high], linestyle='--', linewidth=1.4, color=PALETTE['observed'], alpha=0.9, zorder=2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.grid(True, linestyle='--', linewidth=0.9, color=PALETTE['grid'], alpha=0.9)
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.set_xlabel('Observed AQI', color=PALETTE['text'])
    ax.set_ylabel('Predicted AQI', color=PALETTE['text'])
    display_phase = SET_NAME_MAP.get(phase_name, phase_name)
    ax.set_title(f'DualBranch-CEGNetStyle Regression Fit on {display_phase}', color=PALETTE['text'], pad=12)
    fig.tight_layout()
    save_fig(fig, save_dir, f'dualbranch_scatter_fit_{phase_name.lower()}_fresh')

for phase_name, y_true, dates_phase, key in [('Validation', Y_val_raw, dates_val_final, 'val'), ('Test', Y_test_raw, dates_test_final, 'test')]:
    plot_province_comparison_fresh(phase_name=phase_name, y_true=y_true, dates_phase=dates_phase, key=key, save_dir=SAVE_DIR)
    plot_dualbranch_fit_fresh(phase_name=phase_name, y_true=y_true, y_pred=model_dict_eval['DualBranch-CEGNetStyle'][key], dates_phase=dates_phase, save_dir=SAVE_DIR)
    plot_dualbranch_scatter_fresh(phase_name=phase_name, y_true=y_true, y_pred=model_dict_eval['DualBranch-CEGNetStyle'][key], save_dir=SAVE_DIR)

print('新版小清新论文图已保存到：', SAVE_DIR)
