def make_model_metric_plots():
    metric_cols_show = ['RMSE', 'MAE', 'R2', 'SMAPE', 'EHR_100', 'CSI_100', 'AUC_100', 'EHR_150', 'AUC_150', 'EHR_200', 'AUC_200']
    overall_test = overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].set_index('Model')[metric_cols_show]
    heatmap_plot(overall_test, 'Overall Test Metrics', os.path.join(VIS_DIR, '05_model_metrics', '01_overall_test_metric_heatmap.png'), cmap=CMAP_FRESH, annot=True, fmt='.3f')
    province_test = province_metrics_df[province_metrics_df['Phase'] == 'Test'].set_index('Model')[metric_cols_show]
    heatmap_plot(province_test, 'Provincial Test Metrics', os.path.join(VIS_DIR, '05_model_metrics', '02_province_test_metric_heatmap.png'), cmap=CMAP_MINT, annot=True, fmt='.3f')
    radar_src = overall_test[['RMSE', 'MAE', 'R2', 'EHR_100', 'CSI_100', 'AUC_100']].copy()
    radar_score = normalize_for_radar(radar_src, lower_better_cols=['RMSE', 'MAE'])
    radar_plot(radar_score, 'Comprehensive Model Comparison (Test)', os.path.join(VIS_DIR, '05_model_metrics', '03_model_radar_test.png'))
    city_test_full = city_metrics_df[(city_metrics_df['Model'] == 'DualBranch-CEGNetStyle') & (city_metrics_df['Phase'] == 'Test')].copy()
    city_test_full['City'] = city_test_full['City'].map(city_en)
    city_test_full = city_test_full.set_index('City')[['RMSE', 'MAE', 'R2', 'EHR_100', 'CSI_100', 'AUC_100']]
    heatmap_plot(city_test_full, 'City-level Metrics of DualBranch-CEGNetStyle (Test)', os.path.join(VIS_DIR, '05_model_metrics', '04_full_city_metric_heatmap_test.png'), cmap=CMAP_WARM, annot=True, fmt='.3f')
    rank_targets = {'RMSE': False, 'R2': True, 'EHR_100': True, 'AUC_100': True}
    for metric, desc in rank_targets.items():
        ser = overall_test[metric].sort_values(ascending=not desc)
        lollipop_plot(labels=ser.index.tolist(), values=ser.values.tolist(), title=f'Model Ranking by {metric} (Test)', save_path=os.path.join(VIS_DIR, '05_model_metrics', f'05_rank_{metric}.png'), color='#92A8E8' if desc else '#F3A6B7', xlabel=metric, sort_desc=desc)
