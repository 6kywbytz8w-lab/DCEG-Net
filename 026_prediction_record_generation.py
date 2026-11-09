city_records = []

prov_records = []

for phase, y_true, y_true_poll, dates_phase in [('验证集', Y_val_raw, Y_poll_val, dates_val_final), ('测试集', Y_test_raw, Y_poll_test, dates_test_final)]:
    aqi_rebuild_true, primary_idx_true, _ = china_aqi_from_pollutants_np(y_true_poll)
    for model_name, pred_pack in model_dict_eval.items():
        y_pred = pred_pack['val'] if phase == '验证集' else pred_pack['test']
        for i, city in enumerate(city_order):
            for k, dt in enumerate(dates_phase):
                city_records.append({'Phase': phase, 'Model': model_name, 'Date': dt, 'City': city, 'True_AQI': float(y_true[k, i]), 'Pred_AQI': float(y_pred[k, i]), 'True_AQI_Rebuild_FromPollutants': float(aqi_rebuild_true[k, i]), 'True_PrimaryPollutant': POLLUTANT_COLS[int(primary_idx_true[k, i])]})
        y_true_prov = province_series(y_true)
        y_pred_prov = province_series(y_pred)
        for dt, yt, yp in zip(dates_phase, y_true_prov, y_pred_prov):
            prov_records.append({'Phase': phase, 'Model': model_name, 'Date': dt, 'True_Prov_AQI': float(yt), 'Pred_Prov_AQI': float(yp)})

city_pred_df = pd.DataFrame(city_records)

prov_pred_df = pd.DataFrame(prov_records)

result_excel_path = os.path.join(SAVE_DIR, 'full_results_tables_split.xlsx')

with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
    save_grouped_sheets(df=overall_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='overall', sort_cols=['Phase', 'RMSE'])
    save_grouped_sheets(df=province_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='province', sort_cols=['Phase', 'RMSE'])
    save_grouped_sheets(df=city_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='city', sort_cols=['Phase', 'City'])
    save_grouped_sheets(df=consistency_df, id_cols=['Model', 'Phase'], groups=CONSISTENCY_GROUPS, writer=writer, prefix='consistency', sort_cols=['Phase', 'Pred_vs_RebuildTrueAQI_RMSE'])
    save_grouped_sheets(df=primary_pollutant_df, id_cols=['Model', 'Phase', 'PrimaryPollutant'], groups=PRIMARY_POLLUTANT_GROUPS, writer=writer, prefix='primary_pollutant', sort_cols=['Phase', 'PrimaryPollutant', 'RMSE_TrueAQI'])
    save_grouped_sheets(df=ablation_overall_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, writer=writer, prefix='ablation_overall', sort_cols=['Phase', 'RMSE'])
    save_grouped_sheets(df=ablation_province_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, writer=writer, prefix='ablation_province', sort_cols=['Phase', 'RMSE'])
    save_grouped_sheets(df=ablation_city_df, id_cols=['Variant', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='ablation_city', sort_cols=['Phase', 'City'])
    overall_metrics_df.to_excel(writer, sheet_name='overall_all', index=False)
    province_metrics_df.to_excel(writer, sheet_name='province_all', index=False)
    consistency_df.to_excel(writer, sheet_name='consistency_all', index=False)
    primary_pollutant_df.to_excel(writer, sheet_name='primary_pollutant_all', index=False)
    ablation_overall_df.to_excel(writer, sheet_name='ablation_overall_all', index=False)
    ablation_province_df.to_excel(writer, sheet_name='ablation_province_all', index=False)
    ablation_best_params_df.to_excel(writer, sheet_name='ablation_best_params', index=False)
    overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('RMSE').to_excel(writer, sheet_name='test_rmse_rank_overall', index=False)
    province_metrics_df[province_metrics_df['Phase'] == 'Test'].sort_values('RMSE').to_excel(writer, sheet_name='test_rmse_rank_province', index=False)
    consistency_df[consistency_df['Phase'] == 'Test'].sort_values('Pred_vs_RebuildTrueAQI_RMSE').to_excel(writer, sheet_name='test_consistency_rank', index=False)
    ablation_test_rmse_rank.to_excel(writer, sheet_name='ablation_test_rmse_rank', index=False)
    ablation_test_r2_rank.to_excel(writer, sheet_name='ablation_test_r2_rank', index=False)
    ablation_test_ehr100_rank.to_excel(writer, sheet_name='ablation_test_ehr100_rank', index=False)
    city_pred_df.to_excel(writer, sheet_name='city_predictions', index=False)
    prov_pred_df.to_excel(writer, sheet_name='province_predictions', index=False)
    fusion_coef_df.to_excel(writer, sheet_name='fusion_coefficients', index=False)
    fusion_bias_df.to_excel(writer, sheet_name='fusion_city_bias', index=False)
    weight_df.to_excel(writer, sheet_name='province_weights', index=False)
    main_bias_df.to_excel(writer, sheet_name='main_model_city_bias', index=False)
    mrmr_rank_df.to_excel(writer, sheet_name='mrmr_ranking', index=False)
    stability_df.to_excel(writer, sheet_name='stability_selection', index=False)
    surrogate_metrics_df.to_excel(writer, sheet_name='fs_surrogate_metrics', index=False)
    pd.DataFrame({'Selected_Features': selected_features}).to_excel(writer, sheet_name='selected_features', index=False)
    selection_source_df.to_excel(writer, sheet_name='selected_feature_sources', index=False)
    component_summary_df.to_excel(writer, sheet_name='component_summary', index=False)
    pd.DataFrame(dist_mat, index=city_order, columns=city_order).to_excel(writer, sheet_name='static_dist_matrix')
    pd.DataFrame(A_static_raw, index=city_order, columns=city_order).to_excel(writer, sheet_name='static_adj_raw')
    pd.DataFrame(A_static_rw, index=city_order, columns=city_order).to_excel(writer, sheet_name='static_adj_rw')
