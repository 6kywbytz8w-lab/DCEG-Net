def city_bias_correction(y_val_true, pred_val, pred_test, method='median'):
    if method == 'median':
        bias = np.median(y_val_true - pred_val, axis=0)
    else:
        bias = np.mean(y_val_true - pred_val, axis=0)
    pred_val_bc = np.clip(pred_val + bias.reshape(1, -1), AQI_MIN, AQI_MAX).astype(np.float32)
    pred_test_bc = np.clip(pred_test + bias.reshape(1, -1), AQI_MIN, AQI_MAX).astype(np.float32)
    bias_df = pd.DataFrame({'City': city_order, 'Bias': bias})
    return (pred_val_bc, pred_test_bc, bias_df)

def prepare_ablation_inputs(variant_name):
    Xc_train_v, Xe_train_v, A_train_v = (Xc_train.copy(), Xe_train.copy(), A_train.copy())
    Xc_val_v, Xe_val_v, A_val_v = (Xc_val.copy(), Xe_val.copy(), A_val.copy())
    Xc_test_v, Xe_test_v, A_test_v = (Xc_test.copy(), Xe_test.copy(), A_test.copy())
    if variant_name == 'w/o CEEMDAN Branch':
        Xc_train_v = np.zeros_like(Xc_train_v, dtype=np.float32)
        Xc_val_v = np.zeros_like(Xc_val_v, dtype=np.float32)
        Xc_test_v = np.zeros_like(Xc_test_v, dtype=np.float32)
    elif variant_name == 'w/o Exogenous Branch':
        Xe_train_v = np.zeros_like(Xe_train_v, dtype=np.float32)
        Xe_val_v = np.zeros_like(Xe_val_v, dtype=np.float32)
        Xe_test_v = np.zeros_like(Xe_test_v, dtype=np.float32)
    return (Xc_train_v, Xe_train_v, A_train_v, Xc_val_v, Xe_val_v, A_val_v, Xc_test_v, Xe_test_v, A_test_v)
