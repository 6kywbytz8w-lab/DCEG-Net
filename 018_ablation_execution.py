ablation_results = {}

ablation_pred_val = {}

ablation_pred_test = {}

ablation_param_rows = []

variants_to_run = ABLATION_VARIANTS if RUN_ABLATION else ['Full']

for variant in variants_to_run:
    print(f'\n\n================ 开始运行：{variant} ================\n')
    Xc_train_v, Xe_train_v, A_train_v, Xc_val_v, Xe_val_v, A_val_v, Xc_test_v, Xe_test_v, A_test_v = prepare_ablation_inputs(variant)
    result_v = fit_predict_ensemble(model_name=f'Ablation-{variant}', Xc_train=Xc_train_v, Xe_train=Xe_train_v, Y_train_std=Y_train_std, A_train=A_train_v, Xc_val=Xc_val_v, Xe_val=Xe_val_v, A_val=A_val_v, Xc_test=Xc_test_v, Xe_test=Xe_test_v, A_test=A_test_v, ablation_variant=variant)
    pred_val_v = np.clip(inverse_transform(result_v['val_pred_std'], y_center, y_scale), AQI_MIN, AQI_MAX).astype(np.float32)
    pred_test_v = np.clip(inverse_transform(result_v['test_pred_std'], y_center, y_scale), AQI_MIN, AQI_MAX).astype(np.float32)
    pred_val_v, pred_test_v, bias_df_v = city_bias_correction(Y_val_raw, pred_val_v, pred_test_v, method='median')
    bias_df_v.to_excel(os.path.join(SAVE_DIR, f'ablation_bias_{safe_filename(variant)}.xlsx'), index=False)
    ablation_results[variant] = result_v
    ablation_pred_val[variant] = pred_val_v
    ablation_pred_test[variant] = pred_test_v
    ablation_param_rows.append({'Variant': variant, **result_v['best_params'], 'best_epoch': result_v['best_epoch']})

ablation_best_params_df = pd.DataFrame(ablation_param_rows)

ablation_best_params_df.to_excel(os.path.join(SAVE_DIR, 'ablation_best_params.xlsx'), index=False)

main_result = ablation_results['Full']

pred_val = ablation_pred_val['Full']

pred_test = ablation_pred_test['Full']

main_bias_df = pd.read_excel(os.path.join(SAVE_DIR, f"ablation_bias_{safe_filename('Full')}.xlsx"))

main_bias_df.to_excel(os.path.join(SAVE_DIR, 'main_model_city_bias.xlsx'), index=False)
