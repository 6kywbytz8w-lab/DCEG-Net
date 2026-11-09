plot_model_order = ['DualBranch-CEGNetStyle', 'Fusion', 'Persistence']

for phase_name, y_true, dates_phase, key in [('Val', Y_val_raw, dates_val_final, 'val'), ('Test', Y_test_raw, dates_test_final, 'test')]:
    plt.figure(figsize=(14, 6))
    plt.plot(dates_phase, province_series(y_true), label='True Province AQI', linewidth=2.5)
    for model_name in plot_model_order:
        plt.plot(dates_phase, province_series(model_dict_eval[model_name][key]), label=model_name)
    plt.title(f'Province Level Comparison ({phase_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'province_comparison_{phase_name.lower()}.png'), dpi=220)
    plt.show()

plt.figure(figsize=(10, 5))

plt.bar(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('RMSE')['Model'], overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('RMSE')['RMSE'])

plt.xticks(rotation=20)

plt.ylabel('RMSE')

plt.title('Overall Test RMSE Ranking')

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, 'overall_test_rmse_bar.png'), dpi=220)

plt.show()

pp_plot = primary_pollutant_df[primary_pollutant_df['Phase'] == 'Test'].copy()

if not pp_plot.empty:
    for pollutant in pp_plot['PrimaryPollutant'].unique():
        sub = pp_plot[pp_plot['PrimaryPollutant'] == pollutant].copy()
        plt.figure(figsize=(8, 4))
        plt.bar(sub['Model'], sub['RMSE_TrueAQI'])
        plt.xticks(rotation=20)
        plt.ylabel('RMSE')
        plt.title(f'Test AQI RMSE | True Primary Pollutant = {pollutant}')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'primary_pollutant_{safe_filename(pollutant)}_test_rmse_bar.png'), dpi=220)
        plt.show()

plt.figure(figsize=(12, 6))

plt.bar(ablation_test_rmse_rank['Variant'], ablation_test_rmse_rank['RMSE'])

plt.xticks(rotation=30, ha='right')

plt.ylabel('RMSE')

plt.title('Ablation Study - Test RMSE Ranking')

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, 'ablation_test_rmse_ranking.png'), dpi=220)

plt.show()

plt.figure(figsize=(12, 6))

plt.bar(ablation_test_ehr100_rank['Variant'], ablation_test_ehr100_rank['EHR_100'])

plt.xticks(rotation=30, ha='right')

plt.ylabel('EHR@100')

plt.title('Ablation Study - Test EHR@100 Ranking')

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, 'ablation_test_ehr100_ranking.png'), dpi=220)

plt.show()

summary_note = {'model_name': 'DualBranch CEG-Net Style', 'prediction_protocol': 'Predictions are the arithmetic mean across ENSEMBLE_SEEDS; do not describe these outputs as a single-seed single-model prediction.', 'ensemble_seeds': ENSEMBLE_SEEDS, 'branch_1': 'AQI CEEMDAN decomposition branch', 'branch_2': 'Selected multivariate exogenous feature branch', 'graph_type': 'static geographical prior + sample-wise causal historical-correlation graph', 'primary_target': 'geographically weighted province-wide urban AQI composite (not official AQI)', 'province_weight_method': 'raw symmetric geographic adjacency node strength', 'ceemdan_method': 'causal rolling trailing-window CEEMDAN', 'use_winsorize': USE_WINSORIZE, 'feature_selector': 'time_safe_mrmr_plus_block_stability_selection', 'event_thresholds': EVENT_THRESHOLDS, 'run_ablation': RUN_ABLATION, 'ablation_variants': ABLATION_VARIANTS, 'main_points': ['AQI 分解分支与外生变量分支独立编码后门控融合', '使用 Cheb 图卷积 + 门控时序卷积，贴近 CEG-Net 风格', '默认关闭极端值裁剪，保留事件峰值信息', '保留多变量特征筛选，避免把噪声变量全部送入模型', '训练损失同时优化回归误差与事件识别能力', '输出 AQI 100/150/200 三个阈值下的事件指标', '主模型结果、消融实验结果、关键污染物一致性分析均按分组表展示与保存']}

with open(os.path.join(SAVE_DIR, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_note, f, ensure_ascii=False, indent=2)

print('\n全部运行完成，结果已保存到：', SAVE_DIR)

print('分组汇总文件：', result_excel_path)
