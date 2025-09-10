import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
import networkx as nx
from math import atan2, degrees
import cv2
from petals_generator import generate_petal_mask_from_rgb

def compute_normalized_metrics(mask, petal_mask=None, img_rgb=None, reference_resolution=1000):
    # Calcular métricas originales
    results = compute_vein_metrics(mask, petal_mask, img_rgb)
    
    # Calcular factor de escala relativo a una resolución de referencia
    height, width = mask.shape
    image_diagonal = np.sqrt(height**2 + width**2)
    scale_factor = reference_resolution / image_diagonal
    
    # Normalizar métricas dependientes del tamaño
    results["Vein Thickness (VT)"] *= scale_factor
    results["Areole Size (AS)"] *= (scale_factor ** 2)
    results["Vein-to-Vein Distance (VVD)"] *= scale_factor
    
    return results

from skimage.graph import route_through_array

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
    
    # Extremos del esqueleto (nodos de grado 1)
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
            if G.degree[current] != 2:  # junction o endpoint
                break
        visited.update(path)
        
        # Si la rama es suficientemente larga -> vena principal
        if len(path) >= min_length:
            vein_count += 1
    
    return vein_count

def compute_adjacent_vein_distances(skeleton, binary_mask, G, n_samples=200):
    """
    Calcula la distancia entre venas adyacentes usando la dirección perpendicular al esqueleto.
    Solo toma puntos entre bifurcaciones (no extremos ni nodos de grado >=3).
    """
    coords = np.array([n for n in G.nodes if G.degree[n] == 2])  # puntos entre bifurcaciones
    if len(coords) == 0:
        return []

    # Muestra aleatoria para eficiencia
    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]

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
        # Vector perpendicular (normalizado)
        perp1 = (int(round(-dx / norm)), int(round(dy / norm)))
        perp2 = (int(round(dx / norm)), int(round(-dy / norm)))

        # Busca en ambas direcciones perpendiculares
        min_dist = None
        for perp in [perp1, perp2]:
            for d in range(1, 50):  # Máximo 50 píxeles de búsqueda
                yy = y + perp[1] * d
                xx = x + perp[0] * d
                if 0 <= yy < binary_mask.shape[0] and 0 <= xx < binary_mask.shape[1]:
                    if binary_mask[yy, xx]:
                        min_dist = d if (min_dist is None or d < min_dist) else min_dist
                        break
                else:
                    break
        if min_dist is not None:
            distances.append(min_dist)
    return distances

def compute_vein_metrics(mask: np.ndarray, petal_mask: np.ndarray = None, img_rgb: np.ndarray = None) -> dict:
    """
    Compute vein network metrics. If petal_mask is provided, vein density is calculated
    as (vein area) / (petal area). Otherwise, uses total image area.
    If img_rgb is provided and petal_mask is None, the petal mask will be generated from img_rgb and combined with the vein mask.
    
    Args:
        mask (np.ndarray): Binary mask of the leaf veins (1 = vein, 0 = background).
        petal_mask (np.ndarray, optional): Binary mask of the petal (1 = petal, 0 = background).
        img_rgb (np.ndarray, optional): RGB image to generate petal mask if not provided.
        
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    # Ensure binary mask
    binary_mask = mask > 0 if mask.dtype != bool else mask

    # --- Petal mask logic ---
    if petal_mask is None and img_rgb is not None:
        petal_mask = generate_petal_mask_from_rgb(img_rgb)
        # Add the vein mask to the petal mask
        petal_mask = np.clip(petal_mask + (binary_mask.astype(np.uint8)), 0, 1)

    # --- 1. Vein Density (VD) ---
    vein_area = np.sum(binary_mask)
    if petal_mask is not None:
        petal_area = np.sum(petal_mask > 0)
        leaf_area = petal_area if petal_area > 0 else binary_mask.size
    else:
        leaf_area = binary_mask.size  # Total pixels in the image
    vein_density = vein_area / leaf_area if leaf_area > 0 else 0
    
    # --- 2. Vein Thickness (VT) ---
    # Distance transform: measures the thickness at each vein pixel
    dist_transform = distance_transform_edt(binary_mask)
    vein_thickness = np.mean(dist_transform[binary_mask]) * 2 if np.any(binary_mask) else 0
    
    # --- 3. Areole Size (AS) & Number of Areoles (NA) ---
    # Invert the mask to find enclosed regions (areoles)
    inverted_mask = ~binary_mask
    labeled_areoles = label(inverted_mask, connectivity=1)  # 4-connectivity to avoid diagonal links
    regions = regionprops(labeled_areoles)

    # Filter small noisy regions (optional, adjust min_size as needed)
    min_areole_size = 10  # pixels

    # --- Lógica para filtrar areolas internas (sin fondo ni borde) ---
    h, w = mask.shape[:2]
    def touches_border(region):
        y, x = region.coords[:, 0], region.coords[:, 1]
        return np.any(y == 0) or np.any(y == h-1) or np.any(x == 0) or np.any(x == w-1)

    valid_regions = [r for r in regions if r.area >= min_areole_size and not touches_border(r)]
    if valid_regions:
        all_valid = [r for r in regions if r.area >= min_areole_size]
        if all_valid:
            largest_region = max(all_valid, key=lambda r: r.area)
        else:
            largest_region = None
    # Calcular áreas de areolas igual que en la visualización (sin fondo ni borde)
    areole_areas = []
    if valid_regions:
        # Encontrar la región más grande (fondo) entre todas las regiones
        all_valid = [r for r in regions if r.area >= min_areole_size]
        if all_valid:
            largest_region = max(all_valid, key=lambda r: r.area)
        else:
            largest_region = None
        for r in valid_regions:
            if r is largest_region:
                continue
            areole_areas.append(r.area)
    num_areoles = len(areole_areas)
    mean_areole_size = np.mean(areole_areas) if areole_areas else 0
    
    # --- 4. Branching Angle (BA) ---
    skeleton = skeletonize(binary_mask, method='zhang')
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
    
    # Detect junctions (nodes with degree >= 3)
    junctions = [node for node in G.nodes if G.degree[node] >= 3]
    branching_angles = []
    
    for node in junctions:
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        
        # Compute angles between all pairs of branches
        vectors = []
        for neighbor in neighbors:
            dy = neighbor[0] - node[0]
            dx = neighbor[1] - node[1]
            angle = degrees(atan2(dy, dx)) % 360
            vectors.append(angle)
        
        # Get smallest angles between adjacent branches
        vectors_sorted = sorted(vectors)
        angles = []
        for i in range(len(vectors_sorted)):
            angle_diff = abs(vectors_sorted[i] - vectors_sorted[(i + 1) % len(vectors_sorted)])
            angle_diff = min(angle_diff, 360 - angle_diff)
            angles.append(angle_diff)
        
        if angles:
            branching_angles.append(np.mean(angles))
    
    mean_branching_angle = np.mean(branching_angles) if branching_angles else 0
    
    # --- 5. Vein-to-Vein Distance (VVD) mejorado ---
    vvd_distances = compute_adjacent_vein_distances(skeleton, binary_mask, G)
    mean_vvd = np.mean(vvd_distances) if vvd_distances else 0

    main_veins = count_main_veins(binary_mask, min_length=5)

    # --- Return Results ---
    results = {
        "Vein Density (VD)": vein_density,
        "Vein Thickness (VT)": vein_thickness,
        "Areole Size (AS)": mean_areole_size,
        "Number of Areoles (NA)": num_areoles,
        "Branching Angle (BA)": mean_branching_angle,
        "Vein-to-Vein Distance (VVD)": mean_vvd,
        "Main Veins (MV)": main_veins
    }
    return results