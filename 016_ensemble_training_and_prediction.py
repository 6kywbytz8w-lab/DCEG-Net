def fit_final_model_full_train(Xc_train, Xe_train, Y_train_std, A_train, best_params, best_epoch, seed=42, ablation_variant='Full'):
    set_global_seed(seed)
    ds = DualBranchAQIDataset(Xc_train, Xe_train, Y_train_std, A_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    peak_thr = float(np.quantile(Y_train_std.reshape(-1), PEAK_Q))
    criterion = MultiObjectiveDualLoss(peak_thr=peak_thr, event_focus_thr=EVENT_FOCUS_THR, w_mse=LOSS_W_MSE, w_huber=LOSS_W_HUBER, w_smape=LOSS_W_SMAPE, w_event=LOSS_W_EVENT, peak_weight=PEAK_WEIGHT, delta=HUBER_DELTA, use_event_loss=ablation_variant != 'w/o Event Loss')
    model = build_model(hidden=best_params['hidden'], dropout=best_params['dropout'], ablation_variant=ablation_variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
    train_loss_list = []
    val_loss_stub = []
    for _ in range(best_epoch):
        tr_loss = train_one_epoch(model, dl, optimizer, criterion)
        train_loss_list.append(tr_loss)
        val_loss_stub.append(tr_loss)
    return (model, train_loss_list, val_loss_stub)

def fit_predict_ensemble(model_name, Xc_train, Xe_train, Y_train_std, A_train, Xc_val, Xe_val, A_val, Xc_test, Xe_test, A_test, ablation_variant='Full'):
    print(f'\n================ 搜索参数：{model_name} ================\n')
    best_params, best_epoch = search_best_params_and_epoch(Xc_train=Xc_train, Xe_train=Xe_train, Y_train_std=Y_train_std, A_train=A_train, trials=OPTUNA_TRIALS, splits=OPTUNA_SPLITS, ablation_variant=ablation_variant)
    print(f'{model_name} Best params:', best_params)
    print(f'{model_name} Best epoch:', best_epoch)
    val_pred_ds = DualBranchAQIDataset(Xc_val, Xe_val, None, A_val)
    test_pred_ds = DualBranchAQIDataset(Xc_test, Xe_test, None, A_test)
    val_pred_loader = DataLoader(val_pred_ds, batch_size=batch_size, shuffle=False)
    test_pred_loader = DataLoader(test_pred_ds, batch_size=batch_size, shuffle=False)
    val_preds_all = []
    test_preds_all = []
    model_save_dir = os.path.join(SAVE_DIR, safe_filename(model_name))
    ensure_dir(model_save_dir)
    for seed in ENSEMBLE_SEEDS:
        print(f'{model_name} | seed={seed}')
        model, train_loss_list, val_loss_list = fit_final_model_full_train(Xc_train=Xc_train, Xe_train=Xe_train, Y_train_std=Y_train_std, A_train=A_train, best_params=best_params, best_epoch=best_epoch, seed=seed, ablation_variant=ablation_variant)
        val_pred_std = predict_only(model, val_pred_loader)
        test_pred_std = predict_only(model, test_pred_loader)
        val_preds_all.append(val_pred_std)
        test_preds_all.append(test_pred_std)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'{safe_filename(model_name)}_seed_{seed}.pth'))
        save_curve(train_loss_list=train_loss_list, val_loss_list=val_loss_list, title=f'{model_name} Training Curve (seed={seed})', save_path=os.path.join(model_save_dir, f'{safe_filename(model_name)}_seed_{seed}_curve.png'))
    val_pred_std_mean = np.mean(np.stack(val_preds_all, axis=0), axis=0)
    test_pred_std_mean = np.mean(np.stack(test_preds_all, axis=0), axis=0)
    pd.DataFrame([best_params]).to_excel(os.path.join(model_save_dir, 'best_params.xlsx'), index=False)
    pd.DataFrame([{'best_epoch': best_epoch}]).to_excel(os.path.join(model_save_dir, 'best_epoch.xlsx'), index=False)
    return {'best_params': best_params, 'best_epoch': best_epoch, 'val_pred_std': val_pred_std_mean, 'test_pred_std': test_pred_std_mean}
