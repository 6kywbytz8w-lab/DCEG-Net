def export_grouped_metric_excel():
    out_path = os.path.join(VIS_DIR, 'grouped_metric_tables_en.xlsx')
    with pd.ExcelWriter(out_path) as writer:
        save_grouped_sheets(df=overall_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='overall', sort_cols=['Phase', 'RMSE'])
        save_grouped_sheets(df=province_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, writer=writer, prefix='province', sort_cols=['Phase', 'RMSE'])
        save_grouped_sheets(df=consistency_df, id_cols=['Model', 'Phase'], groups=CONSISTENCY_GROUPS, writer=writer, prefix='consistency', sort_cols=['Phase', 'Pred_vs_RebuildTrueAQI_RMSE'])
        save_grouped_sheets(df=primary_pollutant_df, id_cols=['Model', 'Phase', 'PrimaryPollutant'], groups=PRIMARY_POLLUTANT_GROUPS, writer=writer, prefix='primary_pollutant', sort_cols=['Phase', 'PrimaryPollutant', 'RMSE_TrueAQI'])
        save_grouped_sheets(df=ablation_overall_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, writer=writer, prefix='ablation_overall', sort_cols=['Phase', 'RMSE'])
        save_grouped_sheets(df=ablation_province_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, writer=writer, prefix='ablation_province', sort_cols=['Phase', 'RMSE'])
    print(f'Grouped metric tables saved to: {out_path}')

if VIS_SWITCH['data_overview']:
    vis_try('data_overview', make_data_overview_plots)

if VIS_SWITCH['feature_selection']:
    vis_try('feature_selection', make_feature_selection_plots)

if VIS_SWITCH['graph_structure'] or FORCE_ADJ_VIS:
    vis_try('graph_structure_mandatory', make_graph_structure_plots)

if VIS_SWITCH['ceemdan_branch']:
    vis_try('ceemdan_branch', make_ceemdan_branch_plots)

if VIS_SWITCH['prediction_fit']:
    vis_try('prediction_fit', make_prediction_fit_plots)

if VIS_SWITCH['model_metrics']:
    vis_try('model_metrics', make_model_metric_plots)

if VIS_SWITCH['ablation']:
    vis_try('ablation', make_ablation_plots)

if VIS_SWITCH['fusion']:
    vis_try('fusion', make_fusion_plots)

if VIS_SWITCH['consistency']:
    vis_try('consistency', make_consistency_plots)

if VIS_SWITCH['grouped_metric_excel']:
    vis_try('grouped_metric_excel', export_grouped_metric_excel)

print(f'\nAll visualizations have been saved to: {VIS_DIR}\n')
