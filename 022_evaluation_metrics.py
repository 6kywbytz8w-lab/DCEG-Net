def city_level_metrics(y_true, y_pred, city_names):
    rows = []
    for i, city in enumerate(city_names):
        met = calc_extended_metrics(y_true[:, i], y_pred[:, i], thresholds=EVENT_THRESHOLDS)
        rows.append({'City': city, **met})
    return pd.DataFrame(rows)

def summarize_metrics_rows(y_true, y_pred, split_name, model_name):
    overall = calc_extended_metrics(y_true.reshape(-1), y_pred.reshape(-1), thresholds=EVENT_THRESHOLDS)
    rows = [{'Model': model_name, 'Phase': split_name, 'City': 'Overall', **overall}]
    city_df = city_level_metrics(y_true, y_pred, city_order)
    city_df.insert(0, 'Phase', split_name)
    city_df.insert(0, 'Model', model_name)
    rows.extend(city_df.to_dict('records'))
    y_true_prov = province_series(y_true)
    y_pred_prov = province_series(y_pred)
    prov_met = calc_extended_metrics(y_true_prov, y_pred_prov, thresholds=EVENT_THRESHOLDS)
    prov_row = {'Model': model_name, 'Phase': split_name, 'City': 'Province', **prov_met}
    return (rows, prov_row)
