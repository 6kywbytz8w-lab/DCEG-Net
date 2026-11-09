ablation_overall_rows = []

ablation_province_rows = []

ablation_city_rows = []

for variant in variants_to_run:
    for phase, y_true, y_pred in [('Val', Y_val_raw, ablation_pred_val[variant]), ('Test', Y_test_raw, ablation_pred_test[variant])]:
        overall_met = calc_extended_metrics(y_true.reshape(-1), y_pred.reshape(-1), thresholds=EVENT_THRESHOLDS)
        ablation_overall_rows.append({'Variant': variant, 'Phase': phase, **overall_met})
        prov_true = province_series(y_true)
        prov_pred = province_series(y_pred)
        prov_met = calc_extended_metrics(prov_true, prov_pred, thresholds=EVENT_THRESHOLDS)
        ablation_province_rows.append({'Variant': variant, 'Phase': phase, **prov_met})
        city_df_tmp = city_level_metrics(y_true, y_pred, city_order)
        city_df_tmp.insert(0, 'Phase', phase)
        city_df_tmp.insert(0, 'Variant', variant)
        ablation_city_rows.append(city_df_tmp)

ablation_overall_df = pd.DataFrame(ablation_overall_rows)

ablation_province_df = pd.DataFrame(ablation_province_rows)

ablation_city_df = pd.concat(ablation_city_rows, axis=0, ignore_index=True)

for df_tmp in [ablation_overall_df, ablation_province_df, ablation_city_df]:
    metric_cols = [c for c in round_cols if c in df_tmp.columns]
    df_tmp[metric_cols] = df_tmp[metric_cols].round(4)

ablation_test_rmse_rank = ablation_overall_df[ablation_overall_df['Phase'] == 'Test'].sort_values('RMSE').reset_index(drop=True)

ablation_test_r2_rank = ablation_overall_df[ablation_overall_df['Phase'] == 'Test'].sort_values('R2', ascending=False).reset_index(drop=True)

ablation_test_ehr100_rank = ablation_overall_df[ablation_overall_df['Phase'] == 'Test'].sort_values('EHR_100', ascending=False).reset_index(drop=True)
