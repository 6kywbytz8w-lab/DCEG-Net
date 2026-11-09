import warnings

import os

import math

import copy

import json

import random

from typing import List, Dict, Tuple

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import radians, sin, cos, asin, sqrt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score

from sklearn.model_selection import TimeSeriesSplit

from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

import optuna

from pathlib import Path

import sys

_PROJECT_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]

_PROJECT_ROOT = next((p for p in _PROJECT_CANDIDATES if (p / 'src' / 'dcegnet').exists()), None)

if _PROJECT_ROOT is None:
    raise FileNotFoundError('Cannot locate the project src/dcegnet directory. Run from the project root or notebooks directory.')

sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from dcegnet.ceemdan import CEEMDANConfig, build_rolling_ceemdan_panel

BASE_SEED = 42

SEARCH_SEED = 42

ENSEMBLE_SEEDS = [42, 52, 62]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)

DATA_PATH_CANDIDATES = [str(_PROJECT_ROOT / 'data' / 'input_dataset.xlsx'), './data/input_dataset.xlsx']

DATA_PATH = None

for p in DATA_PATH_CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError(f'未找到数据文件，请检查路径。候选路径：{DATA_PATH_CANDIDATES}')

SHEET_NAME = 'Sheet1'

SAVE_DIR = str(_PROJECT_ROOT / 'outputs')

os.makedirs(SAVE_DIR, exist_ok=True)

history_len = 14

pred_horizon = 1

train_ratio = 0.7

val_ratio = 0.15

batch_size = 32

max_search_epochs = 35

search_patience = 7

final_max_epochs_cap = 60

MIN_FINAL_EPOCH = 6

OPTUNA_TRIALS = 18

OPTUNA_SPLITS = 4

USE_WINSORIZE = False

EXTREME_CLIP_Q_LOW = 0.01

EXTREME_CLIP_Q_HIGH = 0.99

CEEMDAN_WINDOW_LENGTH = 14

CEEMDAN_MAX_IMFS = 6

CEEMDAN_TRIALS = 100

CEEMDAN_EPSILON = 0.005

CEEMDAN_NOISE_SCALE = 1.0

CEEMDAN_BASE_SEED = BASE_SEED

CEEMDAN_FORCE_RECOMPUTE = False

CAUSAL_CEEMDAN = True

MAX_IMF_KEEP = CEEMDAN_MAX_IMFS

USE_RESIDUE_COMPONENT = True

STATIC_K = 3

STATIC_SELF_LOOP_WEIGHT = 1.0

STATIC_GRAPH_SYMMETRIC = True

DROP_HIGH_CORR = True

CORR_THRESHOLD = 0.98

MIN_STD = 1e-08

MRMR_TOP_PRESELECT = 24

STABILITY_BOOTSTRAPS = 50

STABILITY_BLOCK_RATIO = 0.75

STABILITY_THRESHOLD = 0.6

FINAL_TOP_K_MAX = 20

TARGET_COL = 'AQI'

DATE_COL = '日期'

CITY_COL = 'city'

POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3_8h']

MANDATORY_FEATURES = ['AQI', 'AQI_lag1', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3_8h']

STABILITY_XGB_PARAMS = dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=BASE_SEED, n_jobs=-1, verbosity=0, objective='reg:squarederror')

EVENT_THRESHOLDS = [100, 150, 200]

LOSS_W_MSE = 0.4

LOSS_W_HUBER = 0.25

LOSS_W_SMAPE = 0.15

LOSS_W_EVENT = 0.2

PEAK_Q = 0.85

PEAK_WEIGHT = 2.5

HUBER_DELTA = 1.0

EVENT_FOCUS_THR = 100.0

AQI_MIN = 0.0

AQI_MAX = 500.0

PROVINCE_WEIGHT_METHOD = 'raw_graph_node_strength'

FUSION_ALPHAS = [0.01, 0.1, 1.0, 10.0]

CITY_COORDS = {'长春': (125.324501, 43.886841), '吉林': (126.550727, 43.84559), '四平': (124.350398, 43.175755), '辽源': (125.143755, 42.887394), '通化': (125.936501, 41.728401), '白山': (126.424317, 41.942505), '松原': (124.823608, 45.136049), '白城': (122.836498, 45.618713), '延边': (129.517646, 42.904543)}
