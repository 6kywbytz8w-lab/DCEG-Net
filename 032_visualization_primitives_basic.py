def heatmap_plot(df, title, save_path, cmap=CMAP_FRESH, annot=False, fmt='.3f', vmin=None, vmax=None):
    fig_w = max(7, 0.7 * df.shape[1] + 3)
    fig_h = max(4, 0.45 * df.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(df.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    if annot and df.shape[0] * df.shape[1] <= 180:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                txt = 'NA' if pd.isna(val) else format(val, fmt)
                ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='#38556B')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_edgecolor('#C9DCEB')
    save_fig(fig, save_path)

def lollipop_plot(labels, values, title, save_path, color='#8FD3D1', xlabel='Value', sort_desc=True):
    ser = pd.Series(values, index=labels)
    ser = ser.sort_values(ascending=not sort_desc)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(ser) + 1.8)))
    y = np.arange(len(ser))
    ax.hlines(y=y, xmin=0, xmax=ser.values, color='#DCEEF6', linewidth=3)
    ax.scatter(ser.values, y, s=90, color=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ser.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for yi, v in zip(y, ser.values):
        ax.text(v, yi, f' {v:.4f}', va='center', ha='left', fontsize=9, color='#466178')
    save_fig(fig, save_path)

def polar_bar_plot(labels, values, title, save_path):
    values = np.asarray(values, dtype=float)
    labels = list(labels)
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.86
    colors = [SOFT_COLORS[i % len(SOFT_COLORS)] for i in range(n)]
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.bar(theta, values, width=width, bottom=0.0, color=colors, edgecolor='white', linewidth=1.5, alpha=0.95)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, pad=22)
    ax.grid(color='#E6F1F7', linestyle='--', alpha=0.8)
    for ang, val in zip(theta, values):
        ax.text(ang, val + max(values) * 0.05 + 1e-08, f'{val:.2f}', ha='center', va='center', fontsize=8, color='#4A6175')
    save_fig(fig, save_path)

def bubble_matrix_plot(df, title, save_path, cmap=CMAP_FRESH):
    arr = df.values.astype(float)
    nrow, ncol = arr.shape
    xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))
    flat = arr.reshape(-1)
    abs_flat = np.abs(flat)
    s_min, s_max = (40, 900)
    if np.nanmax(abs_flat) - np.nanmin(abs_flat) < 1e-10:
        sizes = np.full_like(abs_flat, (s_min + s_max) / 2.0)
    else:
        sizes = s_min + (abs_flat - np.nanmin(abs_flat)) / (np.nanmax(abs_flat) - np.nanmin(abs_flat) + 1e-08) * (s_max - s_min)
    fig_w = max(7, 0.65 * ncol + 3)
    fig_h = max(5, 0.55 * nrow + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sc = ax.scatter(xv.reshape(-1), yv.reshape(-1), s=sizes, c=flat, cmap=cmap, alpha=0.9, edgecolors='white', linewidths=1.0)
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_edgecolor('#C9DCEB')
    save_fig(fig, save_path)
