PROTECTED_OBSERVATION_COLS = {TARGET_COL, *POLLUTANT_COLS}

covariate_fill_cols = [c for c in num_cols_raw if c not in PROTECTED_OBSERVATION_COLS]

protected_before = df_raw[list(PROTECTED_OBSERVATION_COLS)].copy()

if covariate_fill_cols:
    df_raw.loc[:, covariate_fill_cols] = df_raw.groupby(CITY_COL, sort=False)[covariate_fill_cols].ffill()

for col in PROTECTED_OBSERVATION_COLS:
    if not df_raw[col].equals(protected_before[col]):
        raise AssertionError(f'受保护的真实观测列被插补或修改：{col}')

missing_after_rows = []

for city, group in df_raw.groupby(CITY_COL, sort=False):
    for col in num_cols_raw:
        missing_after_rows.append({'City': city, 'Feature': col, 'Missing_Count': int(group[col].isna().sum()), 'Total_Count': int(len(group)), 'Missing_Rate': float(group[col].isna().mean()) if len(group) else 0.0})

missing_after_df = pd.DataFrame(missing_after_rows)

missing_after_df.to_excel(os.path.join(SAVE_DIR, 'missingness_after_fill.xlsx'), index=False)

pd.DataFrame([{'Stage': 'Before covariate forward-fill', 'Missing_Cells': int(missing_before_df['Missing_Count'].sum()), 'Audited_Cells': int(missing_before_df['Total_Count'].sum()), 'Missing_Rate_Percent': 100.0 * missing_before_df['Missing_Count'].sum() / max(1, int(missing_before_df['Total_Count'].sum()))}, {'Stage': 'After covariate forward-fill', 'Missing_Cells': int(missing_after_df['Missing_Count'].sum()), 'Audited_Cells': int(missing_after_df['Total_Count'].sum()), 'Missing_Rate_Percent': 100.0 * missing_after_df['Missing_Count'].sum() / max(1, int(missing_after_df['Total_Count'].sum()))}]).to_excel(os.path.join(SAVE_DIR, 'missingness_imputation_summary.xlsx'), index=False)

df_fs = df_raw.copy()

df_fs['AQI_lag1'] = df_fs.groupby(CITY_COL)['AQI'].shift(1)

candidate_pool = raw_numeric_cols.copy()

if 'AQI_lag1' not in candidate_pool:
    candidate_pool.append('AQI_lag1')

target_future_col = f'{TARGET_COL}_tplus{pred_horizon}'

df_fs[target_future_col] = df_fs.groupby(CITY_COL)[TARGET_COL].shift(-pred_horizon)

fs_use_cols = [DATE_COL, CITY_COL, target_future_col] + candidate_pool

df_fs = df_fs[fs_use_cols].copy()

df_fs = df_fs.dropna(subset=[target_future_col] + candidate_pool).reset_index(drop=True)

if df_fs.empty:
    raise ValueError('特征选择数据为空，请检查 AQI_lag1 与未来目标构造。')

print('候选特征数（原始特征 + AQI_lag1）:', len(candidate_pool))

print('特征选择可用样本数:', df_fs.shape[0])

all_fs_dates = sorted(df_raw[DATE_COL].dropna().unique())

train_cutoff_fs = all_fs_dates[int(len(all_fs_dates) * train_ratio)]

val_cutoff_fs = all_fs_dates[int(len(all_fs_dates) * (train_ratio + val_ratio))]

calendar_map = {d: k for k, d in enumerate(all_fs_dates)}

def _target_date_from_origin(d):
    pos = calendar_map.get(d)
    return all_fs_dates[pos + pred_horizon] if pos is not None and pos + pred_horizon < len(all_fs_dates) else pd.NaT

df_fs['__target_date'] = df_fs[DATE_COL].map(_target_date_from_origin)

train_df_fs = df_fs[df_fs['__target_date'] < train_cutoff_fs].copy()

val_df_fs = df_fs[(df_fs['__target_date'] >= train_cutoff_fs) & (df_fs['__target_date'] < val_cutoff_fs)].copy()

test_df_fs = df_fs[df_fs['__target_date'] >= val_cutoff_fs].copy()

candidate_features = candidate_pool.copy()

std_series = train_df_fs[candidate_features].std(numeric_only=True)

non_constant_features = std_series[std_series > MIN_STD].index.tolist()

dropped_constant = sorted(list(set(candidate_features) - set(non_constant_features)))

candidate_features = non_constant_features

dropped_high_corr = []

if DROP_HIGH_CORR and len(candidate_features) > 1:
    corr_mat = train_df_fs[candidate_features].corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
    dropped_high_corr = sorted(to_drop)
    candidate_features = [c for c in candidate_features if c not in to_drop]

print('去常数后特征数:', len(non_constant_features))

print('去高相关后特征数:', len(candidate_features))
