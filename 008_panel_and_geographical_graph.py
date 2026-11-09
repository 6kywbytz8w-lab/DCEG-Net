def build_pivot(df_in, value_col, city_order):
    p = df_in.pivot(index=DATE_COL, columns=CITY_COL, values=value_col).sort_index()
    p = p.reindex(columns=city_order)
    return p

pivot_aqi = build_pivot(df_model, TARGET_COL, city_order).loc[dates]

pollutant_pivots = {}

for p in POLLUTANT_COLS:
    pollutant_pivots[p] = build_pivot(df_model, p, city_order).loc[dates]

coords_arr = [CITY_COORDS[c] for c in city_order]

dist_mat = np.zeros((len(city_order), len(city_order)), dtype=np.float32)

for i in range(len(city_order)):
    lon1, lat1 = coords_arr[i]
    for j in range(len(city_order)):
        lon2, lat2 = coords_arr[j]
        if i == j:
            dist_mat[i, j] = 0.0
        else:
            dist_mat[i, j] = haversine(lon1, lat1, lon2, lat2)

def row_normalize_np(A, eps=1e-08):
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    return (A / row_sum).astype(np.float32)

def build_static_graph(dist_mat, k_static=3, sigma=None, self_loop_weight=1.0, make_symmetric=True):
    non_zero_dist = dist_mat[dist_mat > 0]
    if sigma is None:
        if len(non_zero_dist) == 0:
            raise ValueError('距离矩阵中没有非零距离，无法构建静态图。')
        sigma = np.median(non_zero_dist)
    W_geo = np.exp(-dist_mat / (sigma + 1e-08)).astype(np.float32)
    np.fill_diagonal(W_geo, 0.0)
    A_raw = np.zeros_like(W_geo, dtype=np.float32)
    for i in range(W_geo.shape[0]):
        nbrs = np.argsort(W_geo[i])[::-1][:k_static]
        A_raw[i, nbrs] = W_geo[i, nbrs]
    if make_symmetric:
        A_raw = np.maximum(A_raw, A_raw.T).astype(np.float32)
    A_rw = A_raw.copy()
    np.fill_diagonal(A_rw, np.diag(A_rw) + self_loop_weight)
    A_rw = row_normalize_np(A_rw)
    return (A_raw.astype(np.float32), A_rw.astype(np.float32), float(sigma))

A_static_raw, A_static_rw, sigma_dist = build_static_graph(dist_mat=dist_mat, k_static=STATIC_K, sigma=None, self_loop_weight=STATIC_SELF_LOOP_WEIGHT, make_symmetric=STATIC_GRAPH_SYMMETRIC)

print('Geographical prior graph prepared; sample-wise dynamic adjacency is constructed from past AQI only.')

print('sigma_dist =', sigma_dist)

print('A_static_raw shape:', A_static_raw.shape)

print('A_static_rw shape:', A_static_rw.shape)

pd.DataFrame(dist_mat, index=city_order, columns=city_order).to_excel(os.path.join(SAVE_DIR, 'static_distance_matrix.xlsx'))

pd.DataFrame(A_static_raw, index=city_order, columns=city_order).to_excel(os.path.join(SAVE_DIR, 'static_adjacency_raw.xlsx'))

pd.DataFrame(A_static_rw, index=city_order, columns=city_order).to_excel(os.path.join(SAVE_DIR, 'static_adjacency_row_normalized.xlsx'))
