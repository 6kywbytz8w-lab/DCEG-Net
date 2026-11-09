def winsorize_by_city_train_quantiles(df_in, cols, train_dates, city_col='city', date_col='日期', q_low=0.01, q_high=0.99):
    df_out = df_in.copy()
    stats_rows = []
    train_mask_global = df_out[date_col].isin(train_dates)
    for city in sorted(df_out[city_col].unique()):
        city_mask = df_out[city_col] == city
        city_train_mask = city_mask & train_mask_global
        for col in cols:
            if col not in df_out.columns:
                continue
            series_train = pd.to_numeric(df_out.loc[city_train_mask, col], errors='coerce').dropna()
            if len(series_train) < 10:
                stats_rows.append({'City': city, 'Column': col, 'Lower_Train_Q': np.nan, 'Upper_Train_Q': np.nan, 'Clipped_Low_Count': 0, 'Clipped_High_Count': 0, 'Status': 'Skipped: fewer than 10 non-missing training values'})
                continue
            lower = float(series_train.quantile(q_low))
            upper = float(series_train.quantile(q_high))
            before_vals = pd.to_numeric(df_out.loc[city_mask, col], errors='coerce')
            low_cnt = int((before_vals < lower).sum())
            high_cnt = int((before_vals > upper).sum())
            df_out.loc[city_mask, col] = before_vals.clip(lower, upper)
            stats_rows.append({'City': city, 'Column': col, 'Lower_Train_Q': lower, 'Upper_Train_Q': upper, 'Clipped_Low_Count': low_cnt, 'Clipped_High_Count': high_cnt})
    return (df_out, pd.DataFrame(stats_rows))

def zero_crossing_count(x):
    s = np.sign(x).copy()
    s[s == 0] = 1
    return int(np.sum(s[1:] * s[:-1] < 0))

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = (lon2 - lon1, lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371.0

AQI_BP = np.array([0, 50, 100, 150, 200, 300, 400, 500], dtype=np.float32)

POLLUTANT_BREAKPOINTS = {'PM2.5': {'bp': np.array([0, 35, 75, 115, 150, 250, 350, 500], dtype=np.float32), 'iaqi': AQI_BP.copy()}, 'PM10': {'bp': np.array([0, 50, 150, 250, 350, 420, 500, 600], dtype=np.float32), 'iaqi': AQI_BP.copy()}, 'SO2': {'bp': np.array([0, 50, 150, 475, 800, 1600, 2100, 2620], dtype=np.float32), 'iaqi': AQI_BP.copy()}, 'NO2': {'bp': np.array([0, 40, 80, 180, 280, 565, 750, 940], dtype=np.float32), 'iaqi': AQI_BP.copy()}, 'CO': {'bp': np.array([0, 2, 4, 14, 24, 36, 48, 60], dtype=np.float32), 'iaqi': AQI_BP.copy()}, 'O3_8h': {'bp': np.array([0, 100, 160, 215, 265, 800], dtype=np.float32), 'iaqi': np.array([0, 50, 100, 150, 200, 300], dtype=np.float32)}}

def _piecewise_iaqi_np(cp, bp, iaqi):
    cp = np.maximum(cp, 0.0)
    out = np.zeros_like(cp, dtype=np.float32)
    for i in range(len(bp) - 1):
        lo, hi = (bp[i], bp[i + 1])
        ia_lo, ia_hi = (iaqi[i], iaqi[i + 1])
        if i == len(bp) - 2:
            mask = (cp >= lo) & (cp <= hi)
        else:
            mask = (cp >= lo) & (cp < hi)
        out[mask] = (ia_hi - ia_lo) / (hi - lo + 1e-08) * (cp[mask] - lo) + ia_lo
    out[cp > bp[-1]] = iaqi[-1]
    return out.astype(np.float32)

def china_aqi_from_pollutants_np(poll_true):
    iaqi_list = []
    for j, p in enumerate(POLLUTANT_COLS):
        rule = POLLUTANT_BREAKPOINTS[p]
        iaqi = _piecewise_iaqi_np(poll_true[..., j], rule['bp'], rule['iaqi'])
        iaqi_list.append(iaqi)
    iaqi_stack = np.stack(iaqi_list, axis=-1)
    aqi_rebuild = iaqi_stack.max(axis=-1)
    primary_idx = iaqi_stack.argmax(axis=-1)
    return (aqi_rebuild.astype(np.float32), primary_idx.astype(np.int64), iaqi_stack.astype(np.float32))

def aqi_grade_np(aqi):
    aqi = np.asarray(aqi)
    bins = np.array([0, 50, 100, 150, 200, 300, 500], dtype=np.float32)
    return np.digitize(aqi, bins[1:], right=True)

def expected_calibration_error(y_true_bin, prob, n_bins=10):
    y_true_bin = np.asarray(y_true_bin).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(prob)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
        else:
            mask = (prob >= bins[i]) & (prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true_bin[mask].mean()
        conf = prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)

def event_prob_from_aqi(y_pred, thr, temp=18.0):
    y_pred = np.asarray(y_pred).reshape(-1)
    prob = 1.0 / (1.0 + np.exp(-(y_pred - thr) / temp))
    return np.clip(prob, 1e-06, 1 - 1e-06)

def calc_multithreshold_event_metrics(y_true, y_pred, thresholds=(100, 150, 200), eps=1e-08):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('事件评估包含非有限值')
    out = {}
    for thr in thresholds:
        truth = y_true >= thr
        forecast = y_pred >= thr
        tp = int(np.sum(truth & forecast))
        fp = int(np.sum(~truth & forecast))
        fn = int(np.sum(truth & ~forecast))
        tn = int(np.sum(~truth & ~forecast))
        n = tp + fp + fn + tn
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        far = fp / (fp + tn) if fp + tn else np.nan
        csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
        auc = roc_auc_score(truth.astype(int), y_pred) if truth.any() and (~truth).any() else np.nan
        prauc = average_precision_score(truth.astype(int), y_pred) if truth.any() and (~truth).any() else np.nan
        values = dict(TP=tp, FP=fp, FN=fn, TN=tn, Accuracy=(tp + tn) / n if n else np.nan, EHR=recall, Recall=recall, Precision=precision, F1=f1, FAR=far, CSI=csi, AUC=auc, PRAUC=prauc)
        for k, v in values.items():
            out[f'{k}_{thr}'] = float(v) if not isinstance(v, int) else v
        out[f'Brier_{thr}'] = np.nan
        out[f'ECE_{thr}'] = np.nan
    return out

def calc_extended_metrics(y_true, y_pred, thresholds=(100, 150, 200), eps=1e-08):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    base = calc_metrics(y_true, y_pred, eps=eps)
    mbe = np.mean(y_pred - y_true)
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0
    multi_thr = calc_multithreshold_event_metrics(y_true, y_pred, thresholds=thresholds, eps=eps)
    return {**base, 'MBE': float(mbe), 'SMAPE': float(smape), **multi_thr}
