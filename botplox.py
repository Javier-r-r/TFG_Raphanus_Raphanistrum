
import math
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Cargar ambos CSV
df_gt = pd.read_csv('metrics_ground_truth.csv')
df_pred = pd.read_csv('metrics_summary_2.csv')

# Unificar nombres de imagen para emparejar (sin extensión si es necesario)
def clean_name(name):
    return name.split('.')[0]
df_gt['id'] = df_gt['image_name'].apply(clean_name)
df_pred['id'] = df_pred['image_name'].apply(clean_name)

# Métricas a comparar
metricas = [
    ('Vein Density (VD)', 'Densidad de Venas (VD)', 'píxeles/píxel²'),
    ('Vein Thickness (VT)', 'Grosor de Venas (VT)', 'píxeles'),
    ('Areole Size (AS)', 'Tamaño de Aréolas (AS)', 'píxeles²'),
    ('Number of Areoles (NA)', 'Número de Aréolas (NA)', 'cantidad'),
    ('Branching Angle (BA)', 'Ángulo de Ramificación (BA)', 'grados'),
    ('Vein-to-Vein Distance (VVD)', 'Distancia entre Venas (VVD)', 'píxeles'),
    ('Main Veins (MV)', 'Nº Venas Principales (MV)', 'cantidad')
]

# Para cada métrica, crear boxplot con dos columnas por fila (2 imágenes por fila)
ncols = 2
nmetrics = len(metricas)
nrows = math.ceil(nmetrics / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
fig.suptitle('Comparación de métricas: Ground Truth vs Predicción', fontsize=16, fontweight='bold')

# Normalizar e indexar axes como array plano para simplificar el bucle
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for idx, (col, titulo, ylabel) in enumerate(metricas):
    ax = axes_flat[idx]
    # Emparejar por id
    gt_vals = df_gt.set_index('id')[col].dropna()
    pred_vals = df_pred.set_index('id')[col].dropna()
    # Solo ids presentes en ambos
    ids_comunes = gt_vals.index.intersection(pred_vals.index)
    gt_vals = gt_vals.loc[ids_comunes]
    pred_vals = pred_vals.loc[ids_comunes]
    data = [gt_vals.values, pred_vals.values]
    bp = ax.boxplot(data, patch_artist=True, labels=['Ground Truth', 'Predicción'])
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)

# Ocultar subplots vacíos
total_slots = nrows * ncols
for idx in range(nmetrics, total_slots):
    axes_flat[idx].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('boxplots_metricas.png', dpi=300, bbox_inches='tight')
plt.show()