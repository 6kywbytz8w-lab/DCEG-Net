def make_graph_structure_plots():
    dist_df = pd.DataFrame(dist_mat, index=city_order_en, columns=city_order_en)
    heatmap_plot(dist_df, 'Geographical Distance Matrix (km)', os.path.join(VIS_DIR, '02_graph_structure', '01_distance_heatmap.png'), cmap=CMAP_FRESH, annot=True, fmt='.1f')
    A_raw_df = pd.DataFrame(A_static_raw, index=city_order_en, columns=city_order_en)
    heatmap_plot(A_raw_df, 'Raw Adjacency Matrix', os.path.join(VIS_DIR, '02_graph_structure', '02_adjacency_raw_heatmap.png'), cmap=CMAP_MINT, annot=True, fmt='.3f')
    A_rw_df = pd.DataFrame(A_static_rw, index=city_order_en, columns=city_order_en)
    heatmap_plot(A_rw_df, 'Row-normalized Adjacency Matrix', os.path.join(VIS_DIR, '02_graph_structure', '03_adjacency_rw_heatmap.png'), cmap=CMAP_FRESH, annot=True, fmt='.3f')
    bubble_matrix_plot(A_raw_df, 'Adjacency Matrix Bubble Plot', os.path.join(VIS_DIR, '02_graph_structure', '04_adjacency_bubble_matrix.png'), cmap=CMAP_WARM)
    fig, ax = plt.subplots(figsize=(9, 7))
    max_w = np.max(A_static_raw) if np.max(A_static_raw) > 0 else 1.0
    for i, ci in enumerate(city_order):
        xi, yi = CITY_COORDS[ci]
        for j, cj in enumerate(city_order):
            if i >= j:
                continue
            w = max(A_static_raw[i, j], A_static_raw[j, i])
            if w > 0:
                xj, yj = CITY_COORDS[cj]
                ax.plot([xi, xj], [yi, yj], color='#92A8E8', linewidth=0.8 + 4.2 * w / max_w, alpha=0.32)
    node_sizes = 600 + 2500 * (w_space / (w_space.max() + 1e-08))
    xs = [CITY_COORDS[c][0] for c in city_order]
    ys = [CITY_COORDS[c][1] for c in city_order]
    ax.scatter(xs, ys, s=node_sizes, c=w_space, cmap=CMAP_MINT, edgecolor='white', linewidth=1.8, zorder=3)
    for c in city_order:
        x, y = CITY_COORDS[c]
        ax.text(x, y + 0.08, city_en(c), ha='center', va='bottom', fontsize=10, color='#425C73')
    ax.set_title('Static Spatial Graph of Jilin Province')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    save_fig(fig, os.path.join(VIS_DIR, '02_graph_structure', '05_geo_graph_network.png'))
    weight_df_en = weight_df.copy()
    weight_df_en['City'] = weight_df_en['City'].map(city_en)
    lollipop_plot(labels=weight_df_en['City'].tolist(), values=weight_df_en['Weight_Final'].tolist(), title='Provincial Spatial Weights', save_path=os.path.join(VIS_DIR, '02_graph_structure', '06_province_spatial_weight_lollipop.png'), color='#92A8E8', xlabel='Weight', sort_desc=True)

def make_ceemdan_branch_plots():
    comp_cnt_df = component_summary_df.copy()
    comp_cnt_df['City'] = comp_cnt_df['City'].map(city_en)
    comp_cnt_df = comp_cnt_df.set_index('City')[['Mean_Train_IMF_Count', 'Max_Train_IMF_Count', 'Train_Valid_Windows']]
    heatmap_plot(comp_cnt_df, 'Rolling CEEMDAN IMF Count Summary (Train)', os.path.join(VIS_DIR, '03_ceemdan_branch', '01_component_count_heatmap.png'), cmap=CMAP_MINT, annot=True, fmt='.2f')
    valid_train = component_tensor[CEEMDAN_WINDOW_LENGTH - 1:train_end]
    comp_energy = np.mean(np.square(valid_train), axis=0)
    comp_energy_df = pd.DataFrame(comp_energy, index=city_order_en, columns=component_names_full)
    heatmap_plot(comp_energy_df, 'Mean Squared Amplitude of Rolling Components (Train)', os.path.join(VIS_DIR, '03_ceemdan_branch', '02_component_energy_heatmap_train.png'), cmap=CMAP_WARM, annot=False)
    reconstruction_rows = []
    for city_idx, city in enumerate(city_order):
        series = pivot_aqi.loc[dates, city].values.astype(np.float32)
        comps = city_component_list[city_idx]
        valid = np.arange(len(dates)) >= CEEMDAN_WINDOW_LENGTH - 1
        recon = comps.sum(axis=0)
        recon_rmse = calc_metrics(series[valid], recon[valid])['RMSE']
        max_abs_error = float(np.max(np.abs(series[valid] - recon[valid])))
        reconstruction_rows.append({'City': city_en(city), 'Endpoint_Reconstruction_RMSE': recon_rmse, 'Max_Absolute_Reconstruction_Error': max_abs_error, 'Valid_Rolling_Windows': int(valid.sum())})
    reconstruction_df = pd.DataFrame(reconstruction_rows)
    reconstruction_df.to_excel(os.path.join(SAVE_DIR, 'ceemdan_endpoint_reconstruction_audit.xlsx'), index=False)
    lollipop_plot(labels=reconstruction_df['City'].tolist(), values=reconstruction_df['Endpoint_Reconstruction_RMSE'].tolist(), title='Rolling CEEMDAN Endpoint Reconstruction Error', save_path=os.path.join(VIS_DIR, '03_ceemdan_branch', '03_endpoint_reconstruction_rmse.png'), color='#A8D88C', xlabel='RMSE', sort_desc=False)
