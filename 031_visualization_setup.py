from matplotlib.colors import LinearSegmentedColormap

from matplotlib.lines import Line2D

import matplotlib.dates as mdates

VIS_SWITCH = {'data_overview': True, 'feature_selection': True, 'graph_structure': True, 'ceemdan_branch': True, 'prediction_fit': True, 'model_metrics': True, 'ablation': True, 'fusion': True, 'consistency': True, 'grouped_metric_excel': True}

FORCE_ADJ_VIS = True

VIS_DIR = os.path.join(SAVE_DIR, 'all_visualizations_en')

for sub in ['00_data_overview', '01_feature_selection', '02_graph_structure', '03_ceemdan_branch', '04_prediction_fit', '05_model_metrics', '06_ablation', '07_fusion', '08_consistency']:
    ensure_dir(os.path.join(VIS_DIR, sub))

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.facecolor'] = '#FBFEFF'

plt.rcParams['axes.facecolor'] = '#FBFEFF'

plt.rcParams['axes.edgecolor'] = '#C9DCEB'

plt.rcParams['axes.grid'] = True

plt.rcParams['grid.color'] = '#E6F1F7'

plt.rcParams['grid.alpha'] = 0.75

plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 14

plt.rcParams['axes.labelsize'] = 11

SOFT_COLORS = ['#8FD3D1', '#A8D88C', '#F6C77A', '#F3A6B7', '#92A8E8', '#C7B5EA', '#A7E0C2', '#9BD4F5']

MODEL_COLOR_MAP = {'DualBranch-CEGNetStyle': '#8FD3D1', 'Fusion': '#92A8E8', 'Persistence': '#F6C77A', 'MA7': '#F3A6B7'}

ABLATION_COLOR_MAP = {'Full': '#8FD3D1', 'w/o CEEMDAN Branch': '#F3A6B7', 'w/o Exogenous Branch': '#F6C77A', 'w/o Graph Convolution': '#92A8E8', 'w/o Gated Fusion': '#C7B5EA', 'w/o Skip Connection': '#A7E0C2', 'w/o Event Loss': '#9BD4F5'}

THR_COLOR_MAP = {100: '#F6C77A', 150: '#F3A6B7', 200: '#C7B5EA'}

def make_cmap(colors, name):
    return LinearSegmentedColormap.from_list(name, colors)

CMAP_FRESH = make_cmap(['#F8FCFF', '#E3F7F6', '#BCE9D8', '#8FD3D1', '#92A8E8'], 'fresh')

CMAP_WARM = make_cmap(['#FFFDF8', '#FCEFD4', '#F6D8B5', '#F3A6B7', '#C7B5EA'], 'warm')

CMAP_MINT = make_cmap(['#FCFFFD', '#EAF9F0', '#CFEFD9', '#A8D88C', '#8FD3D1'], 'mint')

def save_fig(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches='tight')
    plt.close(fig)

def vis_try(tag, func):
    try:
        func()
        print(f'[VIS OK] {tag}')
    except Exception as e:
        print(f'[VIS FAIL] {tag}: {e}')
