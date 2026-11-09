df_model = df_raw.copy()

if 'AQI_lag1' not in df_model.columns:
    df_model['AQI_lag1'] = df_model.groupby(CITY_COL)['AQI'].shift(1)

keep_cols_for_model = list(dict.fromkeys([DATE_COL, CITY_COL, TARGET_COL] + POLLUTANT_COLS + selected_features + ['AQI_lag1']))

keep_cols_existing = [c for c in keep_cols_for_model if c in df_model.columns]

df_model = df_model[keep_cols_existing].copy()

required_non_na_cols = list(dict.fromkeys([TARGET_COL] + POLLUTANT_COLS + selected_features))

df_model = df_model.dropna(subset=required_non_na_cols).reset_index(drop=True)

complete_dates = []

for dt, grp in df_model.groupby(DATE_COL):
    cities_here = set(grp[CITY_COL].tolist())
    if len(cities_here) == len(city_order):
        complete_dates.append(dt)

complete_dates = sorted(complete_dates)

df_model = df_model[df_model[DATE_COL].isin(complete_dates)].copy().reset_index(drop=True)

if len(complete_dates) == 0:
    raise ValueError('拼接后的建模日期为空，请检查特征选择结果与数据完整性。')

print('建模完整日期数:', len(complete_dates))

print('建模样本数:', df_model.shape[0])

dates = complete_dates

T_all = len(dates)

train_end = int(T_all * train_ratio)

val_end = int(T_all * (train_ratio + val_ratio))

train_dates = dates[:train_end]

val_dates = dates[train_end:val_end]

test_dates = dates[val_end:]

print(f'主模型 Total dates={T_all}, train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}')
