city_order_en = city_list_en(city_order)

if USE_RESIDUE_COMPONENT and max_comp_total >= 1:
    component_names_full = [f'IMF{i + 1}' for i in range(max_comp_total - 1)] + ['Residue']
else:
    component_names_full = [f'IMF{i + 1}' for i in range(max_comp_total)]

def make_data_overview_plots():
    fig, ax = plt.subplots(figsize=(12.5, 2.2))
    ax.axvspan(pd.to_datetime(train_dates[0]), pd.to_datetime(train_dates[-1]), color='#E8F7F5', alpha=0.95, label='Train')
    ax.axvspan(pd.to_datetime(val_dates[0]), pd.to_datetime(val_dates[-1]), color='#FFF2D8', alpha=0.95, label='Validation')
    ax.axvspan(pd.to_datetime(test_dates[0]), pd.to_datetime(test_dates[-1]), color='#FBE3EA', alpha=0.95, label='Test')
    ax.set_xlim(pd.to_datetime(dates[0]), pd.to_datetime(dates[-1]))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('Temporal Split Overview')
    ax.legend(frameon=False, loc='upper left', ncol=3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    save_fig(fig, os.path.join(VIS_DIR, '00_data_overview', '01_time_split_timeline.png'))
    tmp = df_model[[DATE_COL, CITY_COL, TARGET_COL]].copy()
    tmp['Month'] = tmp[DATE_COL].dt.to_period('M').astype(str)
    monthly_heat = tmp.groupby(['Month', CITY_COL])[TARGET_COL].mean().unstack().reindex(columns=city_order)
    monthly_heat.columns = city_order_en
    heatmap_plot(monthly_heat.T, 'Monthly Mean AQI by City', os.path.join(VIS_DIR, '00_data_overview', '02_city_monthly_aqi_heatmap.png'), cmap=CMAP_WARM, annot=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    data = [Y_train_raw.reshape(-1), Y_val_raw.reshape(-1), Y_test_raw.reshape(-1)]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=False, showmedians=True, showextrema=False)
    for body, c in zip(parts['bodies'], ['#8FD3D1', '#F6C77A', '#F3A6B7']):
        body.set_facecolor(c)
        body.set_edgecolor('white')
        body.set_alpha(0.92)
    parts['cmedians'].set_color('#4C6579')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Train', 'Validation', 'Test'])
    ax.set_ylabel('AQI')
    ax.set_title('AQI Distribution across Data Splits')
    save_fig(fig, os.path.join(VIS_DIR, '00_data_overview', '03_aqi_violin_by_split.png'))

def make_feature_selection_plots():
    show_df = mrmr_rank_df.head(min(20, len(mrmr_rank_df))).copy()
    lollipop_plot(labels=show_df['Feature'].tolist(), values=show_df['mRMR_Score'].tolist(), title='Top Features ranked by mRMR', save_path=os.path.join(VIS_DIR, '01_feature_selection', '01_mrmr_lollipop.png'), color='#8FD3D1', xlabel='mRMR Score', sort_desc=True)
    stable_show = stability_df.head(min(20, len(stability_df))).copy()
    polar_bar_plot(labels=stable_show['Feature'].tolist(), values=stable_show['Stability_Freq'].tolist(), title='Feature Stability Frequency', save_path=os.path.join(VIS_DIR, '01_feature_selection', '02_stability_polar_bar.png'))
    corr_df = train_df_fs[selected_features].corr().round(3)
    heatmap_plot(corr_df, 'Correlation Heatmap of Selected Features', os.path.join(VIS_DIR, '01_feature_selection', '03_selected_feature_corr_heatmap.png'), cmap=CMAP_FRESH, annot=False)
    source_map = selection_source_df.copy()
    source_map['stable_flag'] = source_map['Source'].str.contains('stable').astype(int)
    source_map['mandatory_flag'] = source_map['Source'].str.contains('mandatory').astype(int)
    source_plot = source_map.set_index('Feature')[['stable_flag', 'mandatory_flag']]
    source_plot.columns = ['Stable', 'Mandatory']
    heatmap_plot(source_plot, 'Selection Source of Final Features', os.path.join(VIS_DIR, '01_feature_selection', '04_feature_source_heatmap.png'), cmap=CMAP_MINT, annot=True, fmt='.0f', vmin=0, vmax=1)
