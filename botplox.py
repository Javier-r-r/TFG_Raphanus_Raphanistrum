
"""Generate comparative boxplots between ground-truth and predicted metrics.

This script reads two CSVs containing per-image metrics, aligns rows by
image id (filename without extension) and produces a set of boxplots that
compare each metric between the ground truth and prediction sets. The
resulting figure is saved as ``boxplots_metricas.png`` and displayed.
"""

import math
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


df_gt = pd.read_csv('metrics_ground_truth.csv')
df_pred = pd.read_csv('metrics_summary_2.csv')


def clean_name(name: str) -> str:
    """Return the filename without extension used as a stable image id.

    Args:
        name: The original image filename (may include extension).

    Returns:
        The filename base (text before the first dot).
    """
    return name.split('.')[0]


df_gt['id'] = df_gt['image_name'].apply(clean_name)
df_pred['id'] = df_pred['image_name'].apply(clean_name)


metricas = [
    ('Vein Density (VD)', 'Densidad de Venas (VD)', 'píxeles/píxel²'),
    ('Vein Thickness (VT)', 'Grosor de Venas (VT)', 'píxeles'),
    ('Areole Size (AS)', 'Tamaño de Aréolas (AS)', 'píxeles²'),
    ('Number of Areoles (NA)', 'Número de Aréolas (NA)', 'cantidad'),
    ('Branching Angle (BA)', 'Ángulo de Ramificación (BA)', 'grados'),
    ('Vein-to-Vein Distance (VVD)', 'Distancia entre Venas (VVD)', 'píxeles'),
    ('Main Veins (MV)', 'Nº Venas Principales (MV)', 'cantidad')
]


ncols = 2
nmetrics = len(metricas)
nrows = math.ceil(nmetrics / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
fig.suptitle('Comparación de métricas: Ground Truth vs Predicción', fontsize=16, fontweight='bold')


axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for idx, (col, titulo, ylabel) in enumerate(metricas):
    ax = axes_flat[idx]
    gt_vals = df_gt.set_index('id')[col].dropna()
    pred_vals = df_pred.set_index('id')[col].dropna()
    ids_comunes = gt_vals.index.intersection(pred_vals.index)
    gt_vals = gt_vals.loc[ids_comunes]
    pred_vals = pred_vals.loc[ids_comunes]
    data = [gt_vals.values, pred_vals.values]
    bp = ax.boxplot(data, patch_artist=True, labels=['Ground Truth', 'Predicción'])
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)


total_slots = nrows * ncols
for idx in range(nmetrics, total_slots):
    axes_flat[idx].axis('off')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('boxplots_metricas.png', dpi=300, bbox_inches='tight')
plt.show()