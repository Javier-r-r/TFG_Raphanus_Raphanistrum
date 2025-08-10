import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import networkx as nx
from scipy.spatial.distance import pdist


def compute_metrics_from_mask(mask: np.ndarray) -> dict:
    """
    Compute skeleton-based vein metrics from a binary mask.

    Args:
        mask: 2D array. Values can be {0,1} or {0..255}. Non-zero is treated as foreground.

    Returns:
        dict with:
          - Numero de bifurcaciones
          - Numero de extremos
          - Longitud total del esqueleto
          - Longitud media de ramas
          - Distancia media entre bifurcaciones
          - Desviacion estandar de distancias entre bifurcaciones
    """
    # Ensure binary boolean
    if mask.dtype != bool:
        # Consider anything > 0 as foreground
        binaria = mask > 0
    else:
        binaria = mask

    if binaria.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Skeleton
    esqueleto = skeletonize(binaria)

    # Neighbor counts in 8-connectivity
    kernel = np.ones((3, 3), dtype=np.uint8)
    vecinos_total = convolve(esqueleto.astype(np.uint8), kernel, mode="constant")
    # Subtract the center pixel to count only neighbors
    vecinos = vecinos_total - esqueleto.astype(np.uint8)

    bifurcaciones = (esqueleto & (vecinos > 2))
    extremos = (esqueleto & (vecinos == 1))

    num_bifurcaciones = int(np.count_nonzero(bifurcaciones))
    num_extremos = int(np.count_nonzero(extremos))
    longitud_total = int(np.count_nonzero(esqueleto))  # in pixels

    # Build 8-neighborhood graph for skeleton pixels
    coords = np.column_stack(np.nonzero(esqueleto))  # (N, 2) as (y, x)
    G = nx.Graph()
    # Add nodes explicitly
    for y, x in coords:
        G.add_node((int(y), int(x)))

    # Connect edges for 8-neighbors
    H, W = esqueleto.shape
    for y, x in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and esqueleto[ny, nx_]:
                    G.add_edge((int(y), int(x)), (int(ny), int(nx_)))

    # Extract branches: simple path from endpoints to next junction/endpoint
    ramas = []
    visitados = set()
    for nodo in G.nodes:
        if nodo in visitados:
            continue
        if G.degree[nodo] == 1:  # endpoint
            camino = [nodo]
            visitados.add(nodo)
            actual = nodo
            previo = None
            while True:
                vecinos_n = [n for n in G.neighbors(actual) if n != previo]
                if not vecinos_n:
                    break
                previo = actual
                actual = vecinos_n[0]
                camino.append(actual)
                visitados.add(actual)
                if G.degree[actual] != 2:
                    break
            if len(camino) > 1:
                ramas.append(camino)

    longitudes_ramas = [len(r) for r in ramas]  # in pixels (node count)
    media_rama = float(np.mean(longitudes_ramas)) if longitudes_ramas else 0.0

    coord_bif = np.column_stack(np.nonzero(bifurcaciones))
    if len(coord_bif) > 1:
        dist_bif = pdist(coord_bif.astype(float))
        media_dist_bif = float(np.mean(dist_bif))
        std_dist_bif = float(np.std(dist_bif))
    else:
        media_dist_bif = 0.0
        std_dist_bif = 0.0

    resultados = {
        "Numero de bifurcaciones": num_bifurcaciones,
        "Numero de extremos": num_extremos,
        "Longitud total del esqueleto": longitud_total,
        "Longitud media de ramas": media_rama,
        "Distancia media entre bifurcaciones": media_dist_bif,
        "Desviacion estandar de distancias entre bifurcaciones": std_dist_bif,
    }
    return resultados
