METRIC_GROUPS = {'basic_metrics': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'MBE', 'SMAPE'], 'event_100_metrics': ['EHR_100', 'Precision_100', 'FAR_100', 'CSI_100', 'AUC_100', 'ECE_100'], 'event_150_metrics': ['EHR_150', 'Precision_150', 'FAR_150', 'CSI_150', 'AUC_150', 'ECE_150'], 'event_200_metrics': ['EHR_200', 'Precision_200', 'FAR_200', 'CSI_200', 'AUC_200', 'ECE_200']}

CONSISTENCY_GROUPS = {'consistency_core': ['Pred_vs_TrueAQI_RMSE', 'Pred_vs_TrueAQI_MAE', 'Pred_vs_RebuildTrueAQI_RMSE', 'Pred_vs_RebuildTrueAQI_MAE', 'TrueAQI_vs_RebuildTrueAQI_RMSE', 'TrueAQI_vs_RebuildTrueAQI_MAE', 'AQI_Grade_Acc']}

PRIMARY_POLLUTANT_GROUPS = {'primary_pollutant_metrics': ['Count', 'RMSE_TrueAQI', 'MAE_TrueAQI', 'R2_TrueAQI', 'RMSE_RebuildTrueAQI', 'MAE_RebuildTrueAQI', 'AQI_Grade_Acc']}

def print_grouped_tables(df, id_cols, groups, title, sort_cols=None, ascending=True):
    print(f"\n{'=' * 24} {title} {'=' * 24}\n")
    df_show = df.copy()
    if sort_cols is not None:
        df_show = df_show.sort_values(sort_cols, ascending=ascending)
    for group_name, metric_cols in groups.items():
        use_cols = id_cols + [c for c in metric_cols if c in df_show.columns]
        sub_df = df_show[use_cols].copy()
        print(f'\n----- {group_name} -----\n')
        print(sub_df.to_string(index=False))

def save_grouped_sheets(df, id_cols, groups, writer, prefix, sort_cols=None, ascending=True):
    df_save = df.copy()
    if sort_cols is not None:
        df_save = df_save.sort_values(sort_cols, ascending=ascending)
    for group_name, metric_cols in groups.items():
        use_cols = id_cols + [c for c in metric_cols if c in df_save.columns]
        sub_df = df_save[use_cols].copy()
        sheet_name = f'{prefix}_{group_name}'[:31]
        sub_df.to_excel(writer, sheet_name=sheet_name, index=False)

print_grouped_tables(df=overall_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, title='Overall Metrics', sort_cols=['Phase', 'RMSE'])

print_grouped_tables(df=province_metrics_df, id_cols=['Model', 'Phase', 'City'], groups=METRIC_GROUPS, title='Province Metrics', sort_cols=['Phase', 'RMSE'])

print_grouped_tables(df=consistency_df, id_cols=['Model', 'Phase'], groups=CONSISTENCY_GROUPS, title='Pollutant Consistency Metrics', sort_cols=['Phase', 'Pred_vs_RebuildTrueAQI_RMSE'])

print_grouped_tables(df=primary_pollutant_df, id_cols=['Model', 'Phase', 'PrimaryPollutant'], groups=PRIMARY_POLLUTANT_GROUPS, title='Primary Pollutant Grouped Metrics', sort_cols=['Phase', 'PrimaryPollutant', 'RMSE_TrueAQI'])

print_grouped_tables(df=ablation_overall_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, title='Ablation Overall Metrics', sort_cols=['Phase', 'RMSE'])

print_grouped_tables(df=ablation_province_df, id_cols=['Variant', 'Phase'], groups=METRIC_GROUPS, title='Ablation Province Metrics', sort_cols=['Phase', 'RMSE'])

print('\n================ Test RMSE Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('RMSE').to_string(index=False))

print('\n================ Test RMSE Ranking (Province) ================\n')

print(province_metrics_df[province_metrics_df['Phase'] == 'Test'].sort_values('RMSE').to_string(index=False))

print('\n================ Test EHR@100 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('EHR_100', ascending=False).to_string(index=False))

print('\n================ Test EHR@150 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('EHR_150', ascending=False).to_string(index=False))

print('\n================ Test EHR@200 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('EHR_200', ascending=False).to_string(index=False))

print('\n================ Test AUC@100 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('AUC_100', ascending=False).to_string(index=False))

print('\n================ Test AUC@150 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('AUC_150', ascending=False).to_string(index=False))

print('\n================ Test AUC@200 Ranking (Overall) ================\n')

print(overall_metrics_df[overall_metrics_df['Phase'] == 'Test'].sort_values('AUC_200', ascending=False).to_string(index=False))

print('\n================ Test Consistency Ranking ================\n')

print(consistency_df[consistency_df['Phase'] == 'Test'].sort_values('Pred_vs_RebuildTrueAQI_RMSE').to_string(index=False))

print('\n================ Ablation Test RMSE Ranking ================\n')

print(ablation_test_rmse_rank.to_string(index=False))

print('\n================ Ablation Test R2 Ranking ================\n')

print(ablation_test_r2_rank.to_string(index=False))

print('\n================ Ablation Test EHR@100 Ranking ================\n')

print(ablation_test_ehr100_rank.to_string(index=False))
