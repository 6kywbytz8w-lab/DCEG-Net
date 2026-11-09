def make_ablation_plots():
    abl_cols = ['RMSE', 'MAE', 'R2', 'SMAPE', 'EHR_100', 'CSI_100', 'AUC_100', 'EHR_150', 'AUC_150', 'EHR_200', 'AUC_200']
    abl_test = ablation_overall_df[ablation_overall_df['Phase'] == 'Test'].set_index('Variant')[abl_cols]
    abl_test = abl_test.reindex(variants_to_run)
    heatmap_plot(abl_test, 'Ablation Metrics on Test Set', os.path.join(VIS_DIR, '06_ablation', '01_ablation_overall_heatmap_test.png'), cmap=CMAP_WARM, annot=True, fmt='.3f')
    full_row = abl_test.loc['Full']
    delta_df = abl_test.subtract(full_row, axis=1).drop(index='Full')
    heatmap_plot(delta_df[['RMSE', 'MAE', 'R2', 'EHR_100', 'CSI_100', 'AUC_100']], 'Delta Relative to Full Model (Test)', os.path.join(VIS_DIR, '06_ablation', '02_ablation_delta_vs_full_heatmap.png'), cmap=CMAP_FRESH, annot=True, fmt='.3f')
    ceemdan_compare = abl_test.loc[['Full', 'w/o CEEMDAN Branch'], ['RMSE', 'MAE', 'R2', 'EHR_100', 'CSI_100', 'AUC_100']].copy()
    heatmap_plot(ceemdan_compare, 'Contribution of the CEEMDAN Branch', os.path.join(VIS_DIR, '06_ablation', '03_ceemdan_contribution_heatmap.png'), cmap=CMAP_MINT, annot=True, fmt='.3f')
    abl_city_rmse = ablation_city_df[ablation_city_df['Phase'] == 'Test'].copy()
    abl_city_rmse['City'] = abl_city_rmse['City'].map(city_en)
    abl_city_rmse = abl_city_rmse.pivot(index='Variant', columns='City', values='RMSE')
    abl_city_rmse = abl_city_rmse.reindex(index=variants_to_run, columns=city_order_en)
    heatmap_plot(abl_city_rmse, 'City-level RMSE in Ablation Study (Test)', os.path.join(VIS_DIR, '06_ablation', '04_ablation_city_rmse_heatmap_test.png'), cmap=CMAP_WARM, annot=True, fmt='.3f')

def make_fusion_plots():
    coef_cols = [c for c in fusion_coef_df.columns if c not in ['City', 'Intercept']]
    fusion_coef_heat = fusion_coef_df.copy()
    fusion_coef_heat['City'] = fusion_coef_heat['City'].map(city_en)
    fusion_coef_heat = fusion_coef_heat.set_index('City')[coef_cols]
    heatmap_plot(fusion_coef_heat, 'City-wise Fusion Coefficients', os.path.join(VIS_DIR, '07_fusion', '01_fusion_coefficient_heatmap.png'), cmap=CMAP_FRESH, annot=True, fmt='.3f')
    bias_compare = pd.DataFrame({'MainModel_Bias': main_bias_df.set_index('City')['Bias'], 'Fusion_Bias': fusion_bias_df.set_index('City')['Bias']})
    bias_compare.index = [city_en(x) for x in bias_compare.index]
    dumbbell_plot(bias_compare, 'MainModel_Bias', 'Fusion_Bias', 'City-wise Bias Correction Comparison', os.path.join(VIS_DIR, '07_fusion', '02_bias_dumbbell.png'), left_color='#8FD3D1', right_color='#92A8E8')

def make_consistency_plots():
    cons_cols = ['Pred_vs_TrueAQI_RMSE', 'Pred_vs_TrueAQI_MAE', 'Pred_vs_RebuildTrueAQI_RMSE', 'Pred_vs_RebuildTrueAQI_MAE', 'TrueAQI_vs_RebuildTrueAQI_RMSE', 'TrueAQI_vs_RebuildTrueAQI_MAE', 'AQI_Grade_Acc']
    cons_test = consistency_df[consistency_df['Phase'] == 'Test'].set_index('Model')[cons_cols]
    heatmap_plot(cons_test, 'Pollutant Consistency Metrics (Test)', os.path.join(VIS_DIR, '08_consistency', '01_consistency_heatmap_test.png'), cmap=CMAP_MINT, annot=True, fmt='.3f')
    pp_rmse = primary_pollutant_df[primary_pollutant_df['Phase'] == 'Test'].pivot(index='Model', columns='PrimaryPollutant', values='RMSE_TrueAQI')
    pp_rmse = pp_rmse.reindex(index=list(model_dict_eval.keys()))
    heatmap_plot(pp_rmse, 'RMSE by Primary Pollutant Group (Test)', os.path.join(VIS_DIR, '08_consistency', '02_primary_pollutant_rmse_heatmap_test.png'), cmap=CMAP_WARM, annot=True, fmt='.3f')
    pp_acc = primary_pollutant_df[primary_pollutant_df['Phase'] == 'Test'].pivot(index='Model', columns='PrimaryPollutant', values='AQI_Grade_Acc')
    pp_acc = pp_acc.reindex(index=list(model_dict_eval.keys()))
    heatmap_plot(pp_acc, 'AQI Grade Accuracy by Primary Pollutant Group (Test)', os.path.join(VIS_DIR, '08_consistency', '03_primary_pollutant_gradeacc_heatmap_test.png'), cmap=CMAP_FRESH, annot=True, fmt='.3f')
