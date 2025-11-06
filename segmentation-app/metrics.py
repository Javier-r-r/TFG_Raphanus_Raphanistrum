import numpy as np
import networkx as nx

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from math import atan2, degrees
from petals_generator import generate_petal_mask_from_rgb


# ===============================================================
# MÉTRICAS NORMALIZADAS
# ===============================================================

def compute_normalized_metrics(mask, petal_mask=None, img_rgb=None, reference_resolution=1000):
    """
    Contrasta las métricas del modelo a una resolución de referencia común.
    Escala las métricas dependientes del tamaño (grosor, tamaño de areolas, distancia vena-a-vena)
    para que sean comparables entre imágenes de diferentes resoluciones.
    """
    # Calcular métricas originales
    results = compute_vein_metrics(mask, petal_mask, img_rgb)
    
    # Calcular factor de escala relativo a una resolución de referencia
    height, width = mask.shape
    image_diagonal = np.sqrt(height**2 + width**2)
    scale_factor = reference_resolution / image_diagonal
    
    # Normalizar métricas dependientes del tamaño
    results["Vein Thickness (VT)"] *= scale_factor
    results["Areole Size (AS)"] *= scale_factor 
    results["Vein-to-Vein Distance (VVD)"] *= scale_factor
    
    return results


# ===============================================================
# GROSOR DE LAS VENAS (SOLO PUNTO CENTRAL)
# ===============================================================

def compute_vein_thickness_centerline(binary_mask: np.ndarray):
    """
    Calcula el grosor de la vena usando únicamente los puntos centrales del esqueleto,
    excluyendo bifurcaciones y extremos.

    Método:
    - Se calcula el esqueleto completo de la máscara binaria.
    - Se construye un grafo de la red de venas usando NetworkX.
    - Solo se consideran los nodos con grado == 2 (segmentos intermedios, sin bifurcaciones ni extremos).
    - Se toma la transformada de distancia en esos puntos como radio local y se multiplica por 2.
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0

    # Esqueletizar la máscara
    sk = skeletonize(binary_mask)

    # Construir grafo del esqueleto
    coords = np.column_stack(np.where(sk))
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dy, dx) != (0, 0) and (y + dy, x + dx) in G.nodes:
                    G.add_edge((y, x), (y + dy, x + dx))

    # Seleccionar nodos intermedios (para evitar bifurcaciones)
    mid_points = [n for n in G.nodes if G.degree[n] == 2]
    if not mid_points:
        return np.array([]), 0.0, 0.0, 0.0

    # Transformada de distancia
    dist = distance_transform_edt(binary_mask)

    # Radio local en los puntos del centro
    radii = np.array([dist[y, x] for y, x in mid_points])
    local_thickness = radii * 2.0

    mean_t = float(np.mean(local_thickness)) if local_thickness.size else 0.0

    return mean_t

# ===============================================================
# DISTANCIA ENTRE VENAS ADYACENTES
# ===============================================================

def compute_adjacent_vein_distances(skeleton, binary_mask, G, n_samples=200):
    """
    Calcula la distancia entre venas adyacentes usando la dirección perpendicular al esqueleto.
    Solo toma puntos entre bifurcaciones (no extremos ni nodos de grado >=3).
    """
    coords = np.array([n for n in G.nodes if G.degree[n] == 2]) 
    if len(coords) == 0:
        return []

    # Muestra aleatoria para eficiencia
    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]

    # Etiquetar las componentes una sola vez (para distinguir la propia vena)
    labeled = label(binary_mask, connectivity=1)

    distances = []
    for y, x in coords:
        # Encuentra los dos vecinos para estimar la dirección local
        neighbors = list(G.neighbors((y, x)))
        if len(neighbors) != 2:
            continue
        (y1, x1), (y2, x2) = neighbors
        # Vector tangente a la vena
        dy = y2 - y1
        dx = x2 - x1
        norm = np.hypot(dy, dx)
        if norm == 0:
            continue
        # Vector perpendicular (componentes en coordenadas y,x)
        perp_y1 = int(round(-dx / norm))
        perp_x1 = int(round(dy / norm))
        perp_y2 = -perp_y1
        perp_x2 = -perp_x1

        # etiqueta actual del punto de esqueleto
        current_label = labeled[y, x] if (0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]) else 0

        # Busca en ambas direcciones perpendiculares: exige primero salir de la propia
        # vena (encontrar background) y sólo entonces aceptar el siguiente foreground.
        # Esto evita contar el grosor local (d=1) cuando el primer foreground pertenece
        # a la propia vena.
        min_dist = None
        max_search = 200
        for perp_y, perp_x in [(perp_y1, perp_x1), (perp_y2, perp_x2)]:
            found_background = False
            for d in range(1, max_search + 1):
                yy = int(y + perp_y * d)
                xx = int(x + perp_x * d)
                if not (0 <= yy < binary_mask.shape[0] and 0 <= xx < binary_mask.shape[1]):
                    break
                if not binary_mask[yy, xx]:
                    # Hemos salido de la propia vena
                    found_background = True
                    continue
                # Si todavía no hemos salido de la vena, ignoramos foreground (grosor)
                if not found_background:
                    continue
                # Ahora sí: el primer foreground tras el fondo -> registrar distancia
                min_dist = d if (min_dist is None or d < min_dist) else min_dist
                break
        if min_dist is not None:
            distances.append(min_dist)
    # Si no se encontraron distancias por la búsqueda perpendicular, usar fallback KDTree
    if not distances:
        # Obtener puntos de esqueleto por etiqueta (solo etiquetas > 0)
        sk_mask = (skeleton > 0)
        ys, xs = np.where(sk_mask)
        if ys.size == 0:
            return []
        labels_at_points = labeled[ys, xs]
        unique_labels = np.unique(labels_at_points)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size < 2:
            # no hay varias componentes etiquetadas -> no hay vena distinta
            return []

        # Agrupar coords por etiqueta
        points_by_label = {}
        for lab in unique_labels:
            mask_lab = (labels_at_points == lab)
            pts = np.column_stack((ys[mask_lab], xs[mask_lab]))
            if pts.size:
                points_by_label[lab] = pts

        # Calcular distancia mínima entre cada par de componentes con cKDTree
        labels_list = list(points_by_label.keys())
        for i in range(len(labels_list)):
            li = labels_list[i]
            pts_i = points_by_label[li]
            tree_i = cKDTree(pts_i)
            for j in range(i + 1, len(labels_list)):
                lj = labels_list[j]
                pts_j = points_by_label[lj]
                if pts_j.size == 0:
                    continue
                # query pts_j against tree_i
                dists, _ = tree_i.query(pts_j, k=1)
                min_pair_dist = float(np.min(dists)) if dists.size else None
                if min_pair_dist is not None:
                    distances.append(min_pair_dist)
    return distances


# ===============================================================
# CONTEO DE VENAS PRINCIPALES
# ===============================================================

def count_main_veins(mask, min_length=30):
    """
    Cuenta el número de venas principales en la máscara binaria.
    Usa el esqueleto y filtra por longitud mínima.
    
    Args:
        mask (np.ndarray): máscara binaria de las venas.
        min_length (int): longitud mínima de píxeles para considerar vena principal.
    
    Returns:
        int: número de venas principales detectadas.
    """
    # Esqueletizar
    skeleton = skeletonize(mask > 0)
    
    # Construir grafo
    G = nx.Graph()
    coords = np.column_stack(np.where(skeleton))
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_coord = y + dy, x + dx
                if (ny, nx_coord) in G.nodes:
                    G.add_edge((y, x), (ny, nx_coord))
    
    # Extremos del esqueleto 
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    
    visited = set()
    vein_count = 0
    
    for ep in endpoints:
        if ep in visited:
            continue
        # BFS/DFS para seguir toda la rama hasta que termine
        path = [ep]
        current = ep
        prev = None
        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev]
            if not neighbors:
                break
            prev, current = current, neighbors[0]
            path.append(current)
            if G.degree[current] != 2: 
                break
        visited.update(path)
        
        # Si la rama es suficientemente larga -> vena principal
        if len(path) >= min_length:
            vein_count += 1
    
    return vein_count


# ===============================================================
# CÁLCULO PRINCIPAL DE MÉTRICAS
# ===============================================================

def compute_vein_metrics(mask: np.ndarray, petal_mask: np.ndarray = None, img_rgb: np.ndarray = None) -> dict:
    """
    Calcula métricas de la red de venas. Si se proporciona `petal_mask`, la densidad de venas
    se calcula como (área de venas) / (área del pétalo). En caso contrario, se usa el área total de la imagen.
    Si se proporciona `img_rgb` y `petal_mask` es None, la máscara del pétalo se generará a partir de `img_rgb`
    y se combinará con la máscara de venas.

    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    binary_mask = mask > 0 if mask.dtype != bool else mask
    skeleton = skeletonize(binary_mask, method='zhang')

    # --- 1. Densidad de venas (VD) ---
    vein_area = np.sum(binary_mask)
    if petal_mask is None and img_rgb is not None:
        petal_mask = generate_petal_mask_from_rgb(img_rgb)
        petal_mask = np.clip(petal_mask + (binary_mask.astype(np.uint8)), 0, 1)
    petal_area = np.sum(petal_mask > 0) if petal_mask is not None else binary_mask.size
    vein_density = vein_area / petal_area if petal_area > 0 else 0
    
    # --- 2. Grosor de vena (VT) ---
    vt_mean = float(compute_vein_thickness_centerline(binary_mask))

    # --- 3. Tamaño y número de areolas (AS, NA) ---
    inverted_mask = np.logical_not(binary_mask)
    labeled_areoles = label(inverted_mask, connectivity=1)
    regions = regionprops(labeled_areoles)
    min_areole_size = 10

    # Filtrar regiones internas (no en el borde)
    h, w = mask.shape[:2]
    def touches_border(region):
        y, x = region.coords[:, 0], region.coords[:, 1]
        return np.any(y == 0) or np.any(y == h-1) or np.any(x == 0) or np.any(x == w-1)
    valid_regions = [r for r in regions if r.area >= min_areole_size and not touches_border(r)]

    areole_diameters = []
    if valid_regions:
        all_valid = [r for r in regions if r.area >= min_areole_size]
        if all_valid:
            largest_region = max(valid_regions, key=lambda r: r.area)
        else:
            largest_region = None
        areole_diameters = []
        if valid_regions:
            for r in valid_regions:
                if r is largest_region:
                    continue
                equivalent_diameter = 2 * np.sqrt(r.area / np.pi)
                areole_diameters.append(equivalent_diameter)
    num_areoles = len(areole_diameters)
    mean_areole_size = np.mean(areole_diameters) if areole_diameters else 0
    
    # --- 4. Ángulo de bifurcación (BA) ---
    G = nx.Graph()
    coords = np.column_stack(np.where(skeleton))
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_coord = y + dy, x + dx
                if (ny, nx_coord) in G.nodes:
                    G.add_edge((y, x), (ny, nx_coord))

    junctions = [n for n in G.nodes if G.degree[n] >= 3]
    branching_angles = []
    for node in junctions:
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        vectors = []
        for neighbor in neighbors:
            dy = neighbor[0] - node[0]
            dx = neighbor[1] - node[1]
            angle = degrees(atan2(dy, dx)) % 360
            vectors.append(angle)
        vectors_sorted = sorted(vectors)
        angles = []
        for i in range(len(vectors_sorted)):
            angle_diff = abs(vectors_sorted[i] - vectors_sorted[(i + 1) % len(vectors_sorted)])
            angle_diff = min(angle_diff, 360 - angle_diff)
            angles.append(angle_diff)
        if angles:
            branching_angles.append(np.mean(angles))
    mean_branching_angle = np.mean(branching_angles) if branching_angles else 0
    
    # --- 5. Distancia vena-a-vena (VVD) ---
    vvd_distances = compute_adjacent_vein_distances(skeleton, binary_mask, G)
    mean_vvd = np.mean(vvd_distances) if vvd_distances else 0

    # --- 6. Número de venas principales ---
    main_veins = count_main_veins(binary_mask, min_length=5)

    # --- 7. Devolver resultados ---
    results = {
        "Vein Density (VD)": vein_density,
        "Vein Thickness (VT)": vt_mean,
        "Areole Size (AS)": mean_areole_size,
        "Number of Areoles (NA)": num_areoles,
        "Branching Angle (BA)": mean_branching_angle,
        "Vein-to-Vein Distance (VVD)": mean_vvd,
        "Main Veins (MV)": main_veins
    }
    return results
