def search_best_params_and_epoch(Xc_train, Xe_train, Y_train_std, A_train, trials=26, splits=4, ablation_variant='Full'):
    if len(Xc_train) < 20:
        raise ValueError(f'训练样本过少，无法搜索。当前样本数：{len(Xc_train)}')
    effective_splits = min(splits, len(Xc_train) - 1)
    if effective_splits < 2:
        raise ValueError('TimeSeriesSplit 的 n_splits 至少为 2。')
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    def objective(trial):
        hidden = trial.suggest_categorical('hidden', [64, 128, 256])
        lr = trial.suggest_float('lr', 0.0002, 0.003, log=True)
        wd = trial.suggest_float('wd', 1e-06, 8e-05, log=True)
        dropout = trial.suggest_float('dropout', 0.05, 0.25)
        fold_losses = []
        fold_best_epochs = []
        for tr_idx, va_idx in tscv.split(Xc_train):
            Xc_tr, Xe_tr, Y_tr, A_tr = (Xc_train[tr_idx], Xe_train[tr_idx], Y_train_std[tr_idx], A_train[tr_idx])
            Xc_va, Xe_va, Y_va, A_va = (Xc_train[va_idx], Xe_train[va_idx], Y_train_std[va_idx], A_train[va_idx])
            ds_tr = DualBranchAQIDataset(Xc_tr, Xe_tr, Y_tr, A_tr)
            ds_va = DualBranchAQIDataset(Xc_va, Xe_va, Y_va, A_va)
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)
            dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
            peak_thr = float(np.quantile(Y_tr.reshape(-1), PEAK_Q))
            criterion = MultiObjectiveDualLoss(peak_thr=peak_thr, event_focus_thr=EVENT_FOCUS_THR, w_mse=LOSS_W_MSE, w_huber=LOSS_W_HUBER, w_smape=LOSS_W_SMAPE, w_event=LOSS_W_EVENT, peak_weight=PEAK_WEIGHT, delta=HUBER_DELTA, use_event_loss=ablation_variant != 'w/o Event Loss')
            model = build_model(hidden=hidden, dropout=dropout, ablation_variant=ablation_variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            best_loss = np.inf
            best_epoch = 1
            best_state = None
            wait = 0
            for epoch in range(1, max_search_epochs + 1):
                train_one_epoch(model, dl_tr, optimizer, criterion)
                va_loss, _, _ = evaluate(model, dl_va, criterion)
                if va_loss < best_loss:
                    best_loss = va_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= search_patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            fold_losses.append(best_loss)
            fold_best_epochs.append(best_epoch)
        trial.set_user_attr('best_epoch', int(np.median(fold_best_epochs)))
        return float(np.mean(fold_losses))
    set_global_seed(SEARCH_SEED)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params
    best_epoch = study.best_trial.user_attrs['best_epoch']
    best_epoch = int(min(max(best_epoch, MIN_FINAL_EPOCH), final_max_epochs_cap))
    return (best_params, best_epoch)
