def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(BASE_SEED)

def ensure_dir(path):
    if path is None or path == '':
        return
    os.makedirs(path, exist_ok=True)

def safe_filename(name: str) -> str:
    invalid_chars = '\\\\/:*?"<>|'
    for ch in invalid_chars:
        name = name.replace(ch, '_')
    return str(name)

def calc_metrics(y_true, y_pred, eps=1e-08):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {'MSE': float(mse), 'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape), 'R2': float(r2)}

def fit_robust_scaler(arr, eps=1e-06):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        raise ValueError('Cannot fit a robust scaler on an empty training array.')
    if not np.isfinite(arr).all():
        raise ValueError('Robust-scaler training data contain NaN or infinite values.')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be a finite positive threshold.')
    center = np.median(arr, axis=0, keepdims=True).astype(np.float32)
    q1 = np.percentile(arr, 25, axis=0, keepdims=True).astype(np.float32)
    q3 = np.percentile(arr, 75, axis=0, keepdims=True).astype(np.float32)
    iqr = (q3 - q1).astype(np.float32)
    scale = np.where(iqr > eps, iqr, 1.0).astype(np.float32)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError('Robust-scaler fitting produced invalid center/scale parameters.')
    return (center, scale)

def transform_with_scaler(arr, center, scale):
    return ((arr - center) / scale).astype(np.float32)

def inverse_transform(arr_std, center, scale):
    return (arr_std * scale + center).astype(np.float32)

def save_curve(train_loss_list, val_loss_list, title, save_path):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Val')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
