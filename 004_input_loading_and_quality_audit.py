df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

required_cols = [DATE_COL, CITY_COL, TARGET_COL] + POLLUTANT_COLS

missing_required = [c for c in required_cols if c not in df_raw.columns]

if len(missing_required) > 0:
    raise ValueError(f'数据缺少必要列: {missing_required}')

if df_raw[DATE_COL].isna().any():
    raise ValueError('日期字段存在空值')

if df_raw[CITY_COL].isna().any():
    raise ValueError('城市字段存在空值')

df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors='coerce')

if df_raw[DATE_COL].isna().any():
    raise ValueError('日期字段存在无法解析的值')

df_raw[CITY_COL] = df_raw[CITY_COL].astype(str).str.strip()

if df_raw[CITY_COL].eq('').any():
    raise ValueError('城市字段存在空字符串')

if df_raw.duplicated([CITY_COL, DATE_COL]).any():
    dup = df_raw.loc[df_raw.duplicated([CITY_COL, DATE_COL], keep=False), [CITY_COL, DATE_COL]]
    raise ValueError(f'存在重复的城市-日期记录，示例：\\n{dup.head(10)}')

df_raw = df_raw.sort_values([CITY_COL, DATE_COL], kind='stable').reset_index(drop=True)

city_order = sorted(df_raw[CITY_COL].unique().tolist())

missing_coords = [c for c in city_order if c not in CITY_COORDS]

if missing_coords:
    raise ValueError(f'以下城市缺少坐标信息: {missing_coords}')

print('原始数据形状:', df_raw.shape)

print('城市:', city_order)

print('日期范围:', df_raw[DATE_COL].min(), '->', df_raw[DATE_COL].max())

raw_numeric_cols = [c for c in df_raw.columns if c not in [DATE_COL, CITY_COL] and pd.api.types.is_numeric_dtype(df_raw[c])]

print('原始数值列数量:', len(raw_numeric_cols))

dates_for_clip = sorted(df_raw[DATE_COL].dropna().unique())

train_end_for_clip = int(len(dates_for_clip) * train_ratio)

train_dates_for_clip = set(dates_for_clip[:train_end_for_clip])

if USE_WINSORIZE:
    clip_cols = [c for c in ['AQI'] + POLLUTANT_COLS if c in df_raw.columns]
    df_raw, extreme_clip_stats = winsorize_by_city_train_quantiles(df_in=df_raw, cols=clip_cols, train_dates=train_dates_for_clip, q_low=EXTREME_CLIP_Q_LOW, q_high=EXTREME_CLIP_Q_HIGH)
else:
    extreme_clip_stats = pd.DataFrame(columns=['City', 'Column', 'Lower_Train_Q', 'Upper_Train_Q', 'Clipped_Low_Count', 'Clipped_High_Count'])

extreme_clip_stats.to_excel(os.path.join(SAVE_DIR, 'extreme_value_clip_summary.xlsx'), index=False)

num_cols_raw = [c for c in df_raw.columns if c not in [DATE_COL, CITY_COL] and pd.api.types.is_numeric_dtype(df_raw[c])]

missing_before_rows = []

for city, group in df_raw.groupby(CITY_COL, sort=False):
    for col in num_cols_raw:
        missing_before_rows.append({'City': city, 'Feature': col, 'Missing_Count': int(group[col].isna().sum()), 'Total_Count': int(len(group)), 'Missing_Rate': float(group[col].isna().mean()) if len(group) else 0.0})

missing_before_df = pd.DataFrame(missing_before_rows)

missing_before_df.to_excel(os.path.join(SAVE_DIR, 'missingness_before_fill.xlsx'), index=False)
