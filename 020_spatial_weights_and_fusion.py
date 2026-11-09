def build_spatial_weights_from_static_graph(A_raw):
    A = np.asarray(A_raw, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A_static_raw 必须是方阵')
    if np.any(A < 0) or not np.isfinite(A).all():
        raise ValueError('邻接矩阵含非法权重')
    strength = A.sum(axis=1)
    if strength.sum() <= 0:
        raise ValueError('图节点强度全为零')
    weights = strength / strength.sum()
    assert np.isclose(weights.sum(), 1.0)
    return (weights.astype(np.float32), strength.astype(np.float32))

w_space, node_strength = build_spatial_weights_from_static_graph(A_static_raw)

weight_df = pd.DataFrame({'City': city_order, 'Geographic_Node_Strength': node_strength, 'Weight_Final': w_space}).sort_values('Weight_Final', ascending=False)

weight_df.to_excel(os.path.join(SAVE_DIR, 'province_spatial_weights.xlsx'), index=False)

print('Province weights:', dict(zip(city_order, np.round(w_space, 5))))

print('Sum of weights:', float(w_space.sum()))

def province_series(y_city, weights=w_space):
    arr = np.asarray(y_city, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(weights):
        raise ValueError('省域汇总要求 [日期数, 城市数] 且城市顺序为city_order')
    return arr @ np.asarray(weights, dtype=np.float64)

def fit_citywise_fusion(y_val, val_pred_dict, test_pred_dict, city_order, alpha_candidates):
    best_alpha = None
    best_rmse = np.inf
    best_fused_val = None
    best_fused_test = None
    best_coef_df = None
    model_names = list(val_pred_dict.keys())
    for alpha in alpha_candidates:
        fused_val = np.zeros_like(y_val, dtype=np.float32)
        fused_test = np.zeros_like(next(iter(test_pred_dict.values())), dtype=np.float32)
        coef_rows = []
        for i, city in enumerate(city_order):
            Xv = np.column_stack([val_pred_dict[name][:, i] for name in model_names])
            Xt = np.column_stack([test_pred_dict[name][:, i] for name in model_names])
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(Xv, y_val[:, i])
            fused_val[:, i] = reg.predict(Xv)
            fused_test[:, i] = reg.predict(Xt)
            row = {'City': city, 'Intercept': float(reg.intercept_)}
            for j, name in enumerate(model_names):
                row[name] = float(reg.coef_[j])
            coef_rows.append(row)
        fused_val = np.clip(fused_val, AQI_MIN, AQI_MAX).astype(np.float32)
        fused_test = np.clip(fused_test, AQI_MIN, AQI_MAX).astype(np.float32)
        rmse_val = calc_metrics(y_val.reshape(-1), fused_val.reshape(-1))['RMSE']
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_alpha = alpha
            best_fused_val = fused_val
            best_fused_test = fused_test
            best_coef_df = pd.DataFrame(coef_rows)
    return (best_alpha, best_fused_val, best_fused_test, best_coef_df)

candidate_val_pred = {'dual_branch_cegnet_style': pred_val, 'persistence': pers_val, 'ma7': ma7_val}

candidate_test_pred = {'dual_branch_cegnet_style': pred_test, 'persistence': pers_test, 'ma7': ma7_test}

best_fusion_alpha, fused_val, fused_test, fusion_coef_df = fit_citywise_fusion(y_val=Y_val_raw, val_pred_dict=candidate_val_pred, test_pred_dict=candidate_test_pred, city_order=city_order, alpha_candidates=FUSION_ALPHAS)

fusion_coef_df.to_excel(os.path.join(SAVE_DIR, 'fusion_coefficients.xlsx'), index=False)

print('Best fusion alpha:', best_fusion_alpha)

fused_val, fused_test, fusion_bias_df = city_bias_correction(Y_val_raw, fused_val, fused_test, method='median')

fusion_bias_df.to_excel(os.path.join(SAVE_DIR, 'fusion_city_bias.xlsx'), index=False)
