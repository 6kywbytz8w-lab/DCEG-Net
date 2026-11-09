def make_prediction_fit_plots():

    def province_plot(date_seq, y_true, pred_dict, phase_name, save_name):
        fig, ax = plt.subplots(figsize=(13.5, 5.2))
        dt = pd.to_datetime(date_seq)
        ax.plot(dt, province_series(y_true), color='#4D687D', linewidth=2.5, label='True Provincial AQI')
        for model_name, pred in pred_dict.items():
            ax.plot(dt, province_series(pred), linewidth=2.0, color=MODEL_COLOR_MAP.get(model_name, '#92A8E8'), alpha=0.95, label=model_name)
        for thr, c in THR_COLOR_MAP.items():
            ax.axhline(thr, color=c, linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_title(f'Provincial AQI Fit Comparison ({phase_name})')
        ax.legend(frameon=False, ncol=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        save_fig(fig, os.path.join(VIS_DIR, '04_prediction_fit', save_name))
    province_plot(dates_val_final, Y_val_raw, {k: v['val'] for k, v in model_dict_eval.items()}, 'Validation', '01_province_fit_validation.png')
    province_plot(dates_test_final, Y_test_raw, {k: v['test'] for k, v in model_dict_eval.items()}, 'Test', '02_province_fit_test.png')
    dt_all = pd.to_datetime(list(dates_val_final) + list(dates_test_final))
    split_point = pd.to_datetime(dates_test_final[0])
    for i, city in enumerate(city_order):
        y_true = np.concatenate([Y_val_raw[:, i], Y_test_raw[:, i]])
        y_main = np.concatenate([pred_val[:, i], pred_test[:, i]])
        y_fusion = np.concatenate([fused_val[:, i], fused_test[:, i]])
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.axvspan(dt_all[0], pd.to_datetime(dates_val_final[-1]), color='#FFF6E5', alpha=0.55, label='Validation')
        ax.axvspan(split_point, dt_all[-1], color='#FDEBF1', alpha=0.45, label='Test')
        ax.plot(dt_all, y_true, color='#4D687D', linewidth=2.3, label='True AQI')
        ax.plot(dt_all, y_main, color='#8FD3D1', linewidth=1.9, label='DualBranch-CEGNetStyle')
        ax.plot(dt_all, y_fusion, color='#92A8E8', linewidth=1.9, label='Fusion')
        event_mask = y_true >= 100
        ax.fill_between(dt_all, 0, y_true, where=event_mask, color='#F6C77A', alpha=0.14)
        for thr, c in THR_COLOR_MAP.items():
            ax.axhline(thr, color=c, linestyle='--', linewidth=1.0, alpha=0.75)
        ax.set_title(f'{city_en(city)} AQI Fit Comparison (Validation + Test)')
        ax.legend(frameon=False, ncol=4, loc='upper left')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        save_fig(fig, os.path.join(VIS_DIR, '04_prediction_fit', f'03_city_fit_{safe_filename(city_en(city))}.png'))
    for model_name, pack in model_dict_eval.items():
        yt = Y_test_raw.reshape(-1)
        yp = pack['test'].reshape(-1)
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        hb = ax.hexbin(yt, yp, gridsize=28, cmap=CMAP_FRESH, mincnt=1, linewidths=0.15)
        mx = max(np.max(yt), np.max(yp)) + 5
        ax.plot([0, mx], [0, mx], linestyle='--', linewidth=1.3, color='#F3A6B7')
        met = calc_extended_metrics(yt, yp, thresholds=EVENT_THRESHOLDS)
        ax.set_title(f'{model_name} | Observed vs Predicted Density (Test)')
        ax.set_xlabel('Observed AQI')
        ax.set_ylabel('Predicted AQI')
        ax.text(0.03, 0.97, f"RMSE={met['RMSE']:.2f}\nMAE={met['MAE']:.2f}\nR2={met['R2']:.3f}", transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFFFF', edgecolor='#D6EAF4'))
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_edgecolor('#C9DCEB')
        save_fig(fig, os.path.join(VIS_DIR, '04_prediction_fit', f'04_hexbin_{safe_filename(model_name)}.png'))
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    residual_data = [model_dict_eval[m]['test'].reshape(-1) - Y_test_raw.reshape(-1) for m in model_dict_eval.keys()]
    parts = ax.violinplot(residual_data, positions=np.arange(1, len(residual_data) + 1), showmedians=True, showextrema=False)
    for body, name in zip(parts['bodies'], model_dict_eval.keys()):
        body.set_facecolor(MODEL_COLOR_MAP.get(name, '#8FD3D1'))
        body.set_edgecolor('white')
        body.set_alpha(0.92)
    parts['cmedians'].set_color('#4D687D')
    ax.axhline(0, color='#F3A6B7', linestyle='--', linewidth=1.2)
    ax.set_xticks(np.arange(1, len(residual_data) + 1))
    ax.set_xticklabels(list(model_dict_eval.keys()), rotation=15)
    ax.set_ylabel('Residual = Predicted - True')
    ax.set_title('Residual Distribution on Test Set')
    save_fig(fig, os.path.join(VIS_DIR, '04_prediction_fit', '05_test_residual_violin.png'))
    for thr in EVENT_THRESHOLDS:
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        ax.plot([0, 1], [0, 1], linestyle='--', color='#C7B5EA', linewidth=1.2)
        for model_name, pack in model_dict_eval.items():
            x, y = reliability_curve_from_aqi(Y_test_raw, pack['test'], thr=thr, n_bins=10)
            if len(x) == 0:
                continue
            ax.plot(x, y, marker='o', linewidth=2, markersize=5, color=MODEL_COLOR_MAP.get(model_name, '#8FD3D1'), label=model_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'Reliability Curve (AQI ≥ {thr})')
        ax.legend(frameon=False)
        save_fig(fig, os.path.join(VIS_DIR, '04_prediction_fit', f'06_reliability_thr_{thr}.png'))
