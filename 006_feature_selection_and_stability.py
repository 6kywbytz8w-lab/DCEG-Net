def time_safe_mrmr_select(X: pd.DataFrame, y: pd.Series, top_k: int) -> Tuple[List[str], pd.DataFrame]:
    feature_names = X.columns.tolist()
    relevance = mutual_info_regression(X.values, y.values, random_state=BASE_SEED)
    relevance = pd.Series(relevance, index=feature_names).fillna(0.0)
    corr = X.corr().abs().fillna(0.0)
    selected = []
    records = []
    for step in range(min(top_k, len(feature_names))):
        if step == 0:
            best_feat = relevance.sort_values(ascending=False).index[0]
            best_score = relevance.loc[best_feat]
            avg_red = 0.0
        else:
            remain = [f for f in feature_names if f not in selected]
            best_feat = None
            best_score = -1e+18
            avg_red = None
            for f in remain:
                red = corr.loc[f, selected].mean() if len(selected) > 0 else 0.0
                score = relevance.loc[f] - red
                if score > best_score:
                    best_score = score
                    best_feat = f
                    avg_red = red
        selected.append(best_feat)
        records.append({'Step': step + 1, 'Feature': best_feat, 'Relevance_MI': float(relevance.loc[best_feat]), 'Avg_Redundancy': float(avg_red), 'mRMR_Score': float(best_score)})
    return (selected, pd.DataFrame(records))

def block_subsample_dates(date_array, ratio, rng):
    date_array = np.array(sorted(date_array))
    n = len(date_array)
    block_len = max(8, int(np.ceil(n * ratio)))
    block_len = min(block_len, n)
    start = rng.randint(0, n - block_len + 1)
    return date_array[start:start + block_len]

X_train_fs = train_df_fs[candidate_features].copy()

y_train_fs = train_df_fs[target_future_col].copy()

mrmr_features, mrmr_rank_df = time_safe_mrmr_select(X=X_train_fs, y=y_train_fs, top_k=min(MRMR_TOP_PRESELECT, len(candidate_features)))

mrmr_rank_df.to_excel(os.path.join(SAVE_DIR, 'mrmr_ranking.xlsx'), index=False)

print('mRMR 预筛特征数:', len(mrmr_features))

print('mRMR 预筛特征:', mrmr_features)

rng = np.random.RandomState(BASE_SEED)

stability_counts = pd.Series(0, index=mrmr_features, dtype=np.int32)

train_unique_dates_inner = np.array(sorted(train_df_fs[DATE_COL].unique()))

for _ in range(STABILITY_BOOTSTRAPS):
    sampled_dates = block_subsample_dates(train_unique_dates_inner, STABILITY_BLOCK_RATIO, rng)
    boot_df = train_df_fs[train_df_fs[DATE_COL].isin(sampled_dates)].copy()
    if len(boot_df) < 50:
        continue
    X_boot = boot_df[mrmr_features]
    y_boot = boot_df[target_future_col]
    model = XGBRegressor(**STABILITY_XGB_PARAMS)
    model.fit(X_boot, y_boot)
    imp = pd.Series(model.feature_importances_, index=mrmr_features)
    imp = imp.sort_values(ascending=False)
    keep_k = max(5, int(np.ceil(len(mrmr_features) * 0.4)))
    chosen = imp.head(keep_k).index.tolist()
    stability_counts.loc[chosen] += 1

stability_df = pd.DataFrame({'Feature': mrmr_features, 'Stability_Count': stability_counts.values, 'Stability_Freq': stability_counts.values / max(1, STABILITY_BOOTSTRAPS)}).sort_values(['Stability_Freq', 'Stability_Count'], ascending=False).reset_index(drop=True)

stable_features = stability_df.loc[stability_df['Stability_Freq'] >= STABILITY_THRESHOLD, 'Feature'].tolist()

if len(stable_features) == 0:
    stable_features = stability_df.head(min(10, len(stability_df)))['Feature'].tolist()

mandatory_existing = [f for f in MANDATORY_FEATURES if f in candidate_features]

selected_features = list(dict.fromkeys(stable_features + mandatory_existing))

if len(selected_features) > FINAL_TOP_K_MAX:
    stable_rank = stability_df['Feature'].tolist()
    ordered = []
    for f in mandatory_existing:
        if f not in ordered:
            ordered.append(f)
    for f in stable_rank:
        if f not in ordered:
            ordered.append(f)
    selected_features = ordered[:FINAL_TOP_K_MAX]

selection_source = []

for f in selected_features:
    source = []
    if f in stable_features:
        source.append('stable')
    if f in mandatory_existing:
        source.append('mandatory')
    selection_source.append({'Feature': f, 'Source': '+'.join(source)})

selection_source_df = pd.DataFrame(selection_source)

stability_df['Selected'] = stability_df['Feature'].isin(selected_features)

stability_df.to_excel(os.path.join(SAVE_DIR, 'stability_selection.xlsx'), index=False)

selection_source_df.to_excel(os.path.join(SAVE_DIR, 'selected_feature_sources.xlsx'), index=False)

print('最终选中特征数:', len(selected_features))

print('最终选中特征:', selected_features)

final_fs_model = XGBRegressor(**STABILITY_XGB_PARAMS)

final_fs_model.fit(train_df_fs[selected_features], train_df_fs[target_future_col])

pred_val_fs = final_fs_model.predict(val_df_fs[selected_features])

pred_test_fs = final_fs_model.predict(test_df_fs[selected_features])

surrogate_metrics_df = pd.DataFrame([{'Phase': 'Val', **calc_metrics(val_df_fs[target_future_col], pred_val_fs)}, {'Phase': 'Test', **calc_metrics(test_df_fs[target_future_col], pred_test_fs)}])

surrogate_metrics_df.to_excel(os.path.join(SAVE_DIR, 'surrogate_model_metrics.xlsx'), index=False)

pd.DataFrame({'Selected_Features': selected_features}).to_excel(os.path.join(SAVE_DIR, 'selected_feature_list.xlsx'), index=False)
