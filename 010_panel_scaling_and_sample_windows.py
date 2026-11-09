def build_panel_and_check(df_model_local, col_list, date_list, city_list):
    panels = []
    for col in col_list:
        p = build_pivot(df_model_local, col, city_list).loc[date_list]
        if p.isna().any().any():
            raise ValueError(f'选中特征 {col} 在完整日期集上存在缺失，无法安全建模。')
        panels.append(p.values.astype(np.float32))
    return np.stack(panels, axis=-1)

exo_feature_cols = selected_features.copy()

X_exo_panel = build_panel_and_check(df_model, exo_feature_cols, dates, city_order)

y_panel = pivot_aqi.loc[dates, city_order].values.astype(np.float32)

y_poll_panel = np.stack([pollutant_pivots[p].loc[dates, city_order].values.astype(np.float32) for p in POLLUTANT_COLS], axis=-1)

C_dim = component_tensor.shape[-1]

F_dim = X_exo_panel.shape[-1]

print('X_comp_panel shape:', component_tensor.shape)

print('X_exo_panel shape:', X_exo_panel.shape)

print('y_panel shape:', y_panel.shape)

print('y_poll_panel shape:', y_poll_panel.shape)

comp_train = component_tensor[CEEMDAN_WINDOW_LENGTH - 1:train_end].reshape(-1, C_dim)

comp_center, comp_scale = fit_robust_scaler(comp_train)

component_tensor_std = transform_with_scaler(component_tensor.reshape(-1, C_dim), comp_center, comp_scale).reshape(T_all, len(city_order), C_dim)

exo_train = X_exo_panel[:train_end].reshape(-1, F_dim)

exo_center, exo_scale = fit_robust_scaler(exo_train)

X_exo_panel_std = transform_with_scaler(X_exo_panel.reshape(-1, F_dim), exo_center, exo_scale).reshape(T_all, len(city_order), F_dim)

skip_feature_names = [c for c in ['AQI', 'AQI_lag1'] if c in exo_feature_cols]

skip_feature_idx = [exo_feature_cols.index(c) for c in skip_feature_names]

print('Skip features:', skip_feature_names)

def build_dynamic_graph_from_history(aqi_history, geo_prior=None, top_k=STATIC_K, self_loop_weight=STATIC_SELF_LOOP_WEIGHT):
    H = np.asarray(aqi_history, dtype=np.float32)
    n_city = H.shape[1]
    if H.shape[0] < 3:
        corr = np.zeros((n_city, n_city), dtype=np.float32)
    else:
        corr = np.corrcoef(H.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    W = np.maximum(corr, 0.0)
    np.fill_diagonal(W, 0.0)
    if geo_prior is not None:
        W = 0.7 * W + 0.3 * np.asarray(geo_prior, dtype=np.float32)
        np.fill_diagonal(W, 0.0)
    A = np.zeros_like(W, dtype=np.float32)
    for i in range(n_city):
        order = np.argsort(W[i])[::-1]
        order = [j for j in order if j != i and W[i, j] > 0][:top_k]
        if order:
            A[i, order] = W[i, order]
    A = np.maximum(A, A.T).astype(np.float32)
    np.fill_diagonal(A, self_loop_weight)
    return row_normalize_np(A)

def build_samples(comp_panel_std, exo_panel_std, y_panel_raw, y_poll_panel_raw, dates, history_len, pred_horizon):
    Xc_list, Xe_list, Y_list, Ypoll_list, A_seq_list, date_list = ([], [], [], [], [], [])
    T = len(dates)
    first_feature_index = CEEMDAN_WINDOW_LENGTH - 1
    first_origin_index = max(history_len, first_feature_index + history_len)
    for t in range(first_origin_index, T - pred_horizon + 1):
        target_idx = t + pred_horizon - 1
        Xc_list.append(comp_panel_std[t - history_len:t])
        Xe_list.append(exo_panel_std[t - history_len:t])
        Y_list.append(y_panel_raw[target_idx])
        Ypoll_list.append(y_poll_panel_raw[target_idx])
        A_hist = []
        for h_idx in range(t - history_len, t):
            left = max(0, h_idx - history_len + 1)
            hist = y_panel_raw[left:h_idx + 1]
            A_hist.append(build_dynamic_graph_from_history(hist, geo_prior=A_static_raw))
        A_seq_list.append(np.stack(A_hist).astype(np.float32))
        date_list.append(dates[target_idx])
    Xc_arr = np.stack(Xc_list).astype(np.float32)
    Xe_arr = np.stack(Xe_list).astype(np.float32)
    Y_arr = np.stack(Y_list).astype(np.float32)
    Ypoll_arr = np.stack(Ypoll_list).astype(np.float32)
    A_arr = np.stack(A_seq_list).astype(np.float32)
    return (Xc_arr, Xe_arr, Y_arr, Ypoll_arr, A_arr, date_list)

Xc_all, Xe_all, Y_all_raw, Y_poll_all, A_all, target_dates = build_samples(comp_panel_std=component_tensor_std, exo_panel_std=X_exo_panel_std, y_panel_raw=y_panel, y_poll_panel_raw=y_poll_panel, dates=dates, history_len=history_len, pred_horizon=pred_horizon)

FIRST_TARGET_INDEX = max(history_len, CEEMDAN_WINDOW_LENGTH - 1 + history_len) + pred_horizon - 1

target_train_end = train_end - FIRST_TARGET_INDEX

target_val_end = val_end - FIRST_TARGET_INDEX

if target_train_end <= 0 or target_val_end <= target_train_end:
    raise ValueError('滑动窗口切分后训练/验证样本为空，请检查参数。')

Xc_train = Xc_all[:target_train_end]

Xe_train = Xe_all[:target_train_end]

Y_train_raw = Y_all_raw[:target_train_end]

Y_poll_train = Y_poll_all[:target_train_end]

A_train = A_all[:target_train_end]

dates_train_final = target_dates[:target_train_end]

Xc_val = Xc_all[target_train_end:target_val_end]

Xe_val = Xe_all[target_train_end:target_val_end]

Y_val_raw = Y_all_raw[target_train_end:target_val_end]

Y_poll_val = Y_poll_all[target_train_end:target_val_end]

A_val = A_all[target_train_end:target_val_end]

dates_val_final = target_dates[target_train_end:target_val_end]

Xc_test = Xc_all[target_val_end:]

Xe_test = Xe_all[target_val_end:]

Y_test_raw = Y_all_raw[target_val_end:]

Y_poll_test = Y_poll_all[target_val_end:]

A_test = A_all[target_val_end:]

dates_test_final = target_dates[target_val_end:]

print('Train samples:', len(Xc_train))

print('Val samples:', len(Xc_val))

print('Test samples:', len(Xc_test))

y_center, y_scale = fit_robust_scaler(Y_train_raw)

Y_train_std = transform_with_scaler(Y_train_raw, y_center, y_scale)
