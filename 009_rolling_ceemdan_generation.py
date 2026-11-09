if not USE_RESIDUE_COMPONENT:
    raise RuntimeError('The residue channel is required by the fixed-width component tensor definition.')

if not CAUSAL_CEEMDAN:
    raise RuntimeError('This pipeline requires causal rolling CEEMDAN; non-causal decomposition is disabled.')

ceemdan_config = CEEMDANConfig(window_length=CEEMDAN_WINDOW_LENGTH, max_imfs=CEEMDAN_MAX_IMFS, trials=CEEMDAN_TRIALS, epsilon=CEEMDAN_EPSILON, noise_scale=CEEMDAN_NOISE_SCALE, base_seed=CEEMDAN_BASE_SEED, parallel=False)

component_tensor, city_component_list, city_imf_count_list, component_summary_df, component_endpoint_df = build_rolling_ceemdan_panel(pivot_aqi, dates, city_order, ceemdan_config, train_end_index=train_end, cache_path=os.path.join(SAVE_DIR, 'rolling_ceemdan_cache.npz'), force=CEEMDAN_FORCE_RECOMPUTE, progress=True)

C_dim_fixed = CEEMDAN_MAX_IMFS + int(USE_RESIDUE_COMPONENT)

max_comp_total = C_dim_fixed

component_summary_df.to_excel(os.path.join(SAVE_DIR, 'component_summary.xlsx'), index=False)

component_endpoint_df.to_csv(os.path.join(SAVE_DIR, 'rolling_ceemdan_endpoint_features.csv'), index=False, encoding='utf-8-sig')
