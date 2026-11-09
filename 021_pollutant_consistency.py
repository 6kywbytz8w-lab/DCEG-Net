def overall_pollutant_consistency(y_true_aqi, y_true_poll, y_pred_aqi, phase_name, model_name):
    aqi_rebuild_true, _, _ = china_aqi_from_pollutants_np(y_true_poll)
    pred_vs_true = calc_metrics(y_true_aqi.reshape(-1), y_pred_aqi.reshape(-1))
    pred_vs_rebuild = calc_metrics(aqi_rebuild_true.reshape(-1), y_pred_aqi.reshape(-1))
    data_internal = calc_metrics(y_true_aqi.reshape(-1), aqi_rebuild_true.reshape(-1))
    grade_acc = float((aqi_grade_np(y_true_aqi).reshape(-1) == aqi_grade_np(y_pred_aqi).reshape(-1)).mean())
    return {'Model': model_name, 'Phase': phase_name, 'Pred_vs_TrueAQI_RMSE': pred_vs_true['RMSE'], 'Pred_vs_TrueAQI_MAE': pred_vs_true['MAE'], 'Pred_vs_RebuildTrueAQI_RMSE': pred_vs_rebuild['RMSE'], 'Pred_vs_RebuildTrueAQI_MAE': pred_vs_rebuild['MAE'], 'TrueAQI_vs_RebuildTrueAQI_RMSE': data_internal['RMSE'], 'TrueAQI_vs_RebuildTrueAQI_MAE': data_internal['MAE'], 'AQI_Grade_Acc': grade_acc}

def build_primary_pollutant_tables(y_true_aqi, y_true_poll, y_pred_aqi, phase_name, model_name):
    aqi_rebuild_true, true_primary_idx, _ = china_aqi_from_pollutants_np(y_true_poll)
    rows = []
    for j, p in enumerate(POLLUTANT_COLS):
        mask = true_primary_idx.reshape(-1) == j
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        yt = y_true_aqi.reshape(-1)[mask]
        yp = y_pred_aqi.reshape(-1)[mask]
        yr = aqi_rebuild_true.reshape(-1)[mask]
        met_direct = calc_metrics(yt, yp)
        met_rebuild = calc_metrics(yr, yp)
        grade_acc = float((aqi_grade_np(yt) == aqi_grade_np(yp)).mean())
        rows.append({'Model': model_name, 'Phase': phase_name, 'PrimaryPollutant': p, 'Count': cnt, 'RMSE_TrueAQI': met_direct['RMSE'], 'MAE_TrueAQI': met_direct['MAE'], 'R2_TrueAQI': met_direct['R2'], 'RMSE_RebuildTrueAQI': met_rebuild['RMSE'], 'MAE_RebuildTrueAQI': met_rebuild['MAE'], 'AQI_Grade_Acc': grade_acc})
    return pd.DataFrame(rows)
