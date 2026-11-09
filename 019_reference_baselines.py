def build_persistence_pred(y_panel, history_len, pred_horizon, start_idx, end_idx):
    preds = []
    for idx in range(start_idx, end_idx):
        target_idx = FIRST_TARGET_INDEX + idx
        prev_idx = target_idx - pred_horizon
        preds.append(y_panel[prev_idx])
    return np.stack(preds).astype(np.float32)

def build_ma_pred(y_panel, history_len, pred_horizon, start_idx, end_idx, window=7):
    preds = []
    for idx in range(start_idx, end_idx):
        target_idx = FIRST_TARGET_INDEX + idx
        left = max(0, target_idx - window)
        preds.append(y_panel[left:target_idx].mean(axis=0))
    return np.stack(preds).astype(np.float32)

val_start = target_train_end

val_end_idx = target_val_end

test_start = target_val_end

test_end_idx = len(target_dates)

pers_val = build_persistence_pred(y_panel, history_len, pred_horizon, val_start, val_end_idx)

pers_test = build_persistence_pred(y_panel, history_len, pred_horizon, test_start, test_end_idx)

ma7_val = build_ma_pred(y_panel, history_len, pred_horizon, val_start, val_end_idx, window=7)

ma7_test = build_ma_pred(y_panel, history_len, pred_horizon, test_start, test_end_idx, window=7)
