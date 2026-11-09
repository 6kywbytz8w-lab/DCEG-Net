model_dict_eval = {'DualBranch-CEGNetStyle': {'val': pred_val, 'test': pred_test}, 'Fusion': {'val': fused_val, 'test': fused_test}, 'Persistence': {'val': pers_val, 'test': pers_test}, 'MA7': {'val': ma7_val, 'test': ma7_test}}

overall_rows, province_metric_rows = ([], [])

consistency_rows = []

primary_pollutant_rows = []

for model_name, pred_pack in model_dict_eval.items():
    for phase, y_true_aqi, y_true_poll, y_pred in [('Val', Y_val_raw, Y_poll_val, pred_pack['val']), ('Test', Y_test_raw, Y_poll_test, pred_pack['test'])]:
        rows, prov_row = summarize_metrics_rows(y_true_aqi, y_pred, phase, model_name)
        overall_rows.extend(rows)
        province_metric_rows.append(prov_row)
        consistency_rows.append(overall_pollutant_consistency(y_true_aqi=y_true_aqi, y_true_poll=y_true_poll, y_pred_aqi=y_pred, phase_name=phase, model_name=model_name))
        primary_pollutant_rows.append(build_primary_pollutant_tables(y_true_aqi=y_true_aqi, y_true_poll=y_true_poll, y_pred_aqi=y_pred, phase_name=phase, model_name=model_name))

overall_metrics_full_df = pd.DataFrame(overall_rows)

overall_metrics_df = overall_metrics_full_df[overall_metrics_full_df['City'] == 'Overall'].reset_index(drop=True)

city_metrics_df = overall_metrics_full_df[~overall_metrics_full_df['City'].isin(['Overall', 'Province'])].reset_index(drop=True)

province_metrics_df = pd.DataFrame(province_metric_rows)

consistency_df = pd.DataFrame(consistency_rows)

primary_pollutant_df = pd.concat(primary_pollutant_rows, axis=0, ignore_index=True)

round_cols = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'MBE', 'SMAPE', 'EHR_100', 'Precision_100', 'FAR_100', 'CSI_100', 'AUC_100', 'ECE_100', 'EHR_150', 'Precision_150', 'FAR_150', 'CSI_150', 'AUC_150', 'ECE_150', 'EHR_200', 'Precision_200', 'FAR_200', 'CSI_200', 'AUC_200', 'ECE_200']

for df_tmp in [overall_metrics_df, city_metrics_df, province_metrics_df]:
    metric_cols = [c for c in round_cols if c in df_tmp.columns]
    df_tmp[metric_cols] = df_tmp[metric_cols].round(4)

for c in ['Pred_vs_TrueAQI_RMSE', 'Pred_vs_TrueAQI_MAE', 'Pred_vs_RebuildTrueAQI_RMSE', 'Pred_vs_RebuildTrueAQI_MAE', 'TrueAQI_vs_RebuildTrueAQI_RMSE', 'TrueAQI_vs_RebuildTrueAQI_MAE', 'AQI_Grade_Acc']:
    if c in consistency_df.columns:
        consistency_df[c] = consistency_df[c].round(4)

for c in ['RMSE_TrueAQI', 'MAE_TrueAQI', 'R2_TrueAQI', 'RMSE_RebuildTrueAQI', 'MAE_RebuildTrueAQI', 'AQI_Grade_Acc']:
    if c in primary_pollutant_df.columns:
        primary_pollutant_df[c] = primary_pollutant_df[c].round(4)
