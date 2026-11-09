def radar_plot(score_df, title, save_path):
    labels = score_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)
    for i, (name, row) in enumerate(score_df.iterrows()):
        vals = row.values.astype(float).tolist()
        vals = vals + [vals[0]]
        color = MODEL_COLOR_MAP.get(name, SOFT_COLORS[i % len(SOFT_COLORS)])
        ax.plot(angles, vals, linewidth=2.2, color=color, label=name)
        ax.fill(angles, vals, color=color, alpha=0.16)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), frameon=False)
    save_fig(fig, save_path)

def reliability_curve_from_aqi(y_true, y_pred, thr, n_bins=10):
    y_true_bin = (np.asarray(y_true).reshape(-1) >= thr).astype(int)
    prob = event_prob_from_aqi(np.asarray(y_pred).reshape(-1), thr=thr)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = ([], [])
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
        else:
            mask = (prob >= bins[i]) & (prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        xs.append(prob[mask].mean())
        ys.append(y_true_bin[mask].mean())
    return (np.array(xs), np.array(ys))

def dumbbell_plot(df, left_col, right_col, title, save_path, left_color='#8FD3D1', right_color='#92A8E8'):
    df = df.copy().sort_values(right_col)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.hlines(y, df[left_col], df[right_col], color='#D8EAF3', linewidth=2.5)
    ax.scatter(df[left_col], y, s=90, color=left_color, edgecolor='white', linewidth=1.3, label=left_col, zorder=3)
    ax.scatter(df[right_col], y, s=90, color=right_color, edgecolor='white', linewidth=1.3, label=right_col, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index.tolist())
    ax.set_title(title)
    ax.legend(frameon=False)
    for yi, lv, rv in zip(y, df[left_col].values, df[right_col].values):
        ax.text(lv, yi, f'{lv:.2f} ', ha='right', va='center', fontsize=8, color='#466178')
        ax.text(rv, yi, f' {rv:.2f}', ha='left', va='center', fontsize=8, color='#466178')
    save_fig(fig, save_path)

def normalize_for_radar(df, lower_better_cols):
    out = df.copy().astype(float)
    for col in out.columns:
        arr = out[col].values.astype(float)
        mn, mx = (np.nanmin(arr), np.nanmax(arr))
        if mx - mn < 1e-12:
            out[col] = 1.0
        elif col in lower_better_cols:
            out[col] = (mx - arr) / (mx - mn + 1e-08)
        else:
            out[col] = (arr - mn) / (mx - mn + 1e-08)
    return out
