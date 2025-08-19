import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
import networkx as nx
from math import atan2, degrees
import cv2
from petals_generator import generate_petal_mask_from_rgb

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
    areole_areas = [r.area for r in regions if r.area >= min_areole_size]
    
    num_areoles = len(areole_areas)
    mean_areole_size = np.mean(areole_areas) if areole_areas else 0
    
    # --- 4. Branching Angle (BA) ---
    skeleton = skeletonize(binary_mask)
    G = nx.Graph()
    coords = np.column_stack(np.where(skeleton))
    
    # Build graph from skeleton
    for y, x in coords:
        G.add_node((y, x))
    
    # Connect adjacent pixels (8-connectivity)
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_coord = y + dy, x + dx  # <- Cambia 'nx' a 'nx_coord'
                if (ny, nx_coord) in G.nodes:  # <- Usa 'nx_coord' aquí
                    G.add_edge((y, x), (ny, nx_coord))  # <- Y aquí
    
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
    
    # --- 5. Vein-to-Vein Distance (VVD) ---
    # Approximate by computing the distance between skeleton pixels
    if len(coords) >= 2:
        random_samples = min(1000, len(coords))  # Limit to 1000 random points for speed
        sampled_coords = coords[np.random.choice(len(coords), random_samples, replace=False)]
        pairwise_distances = distance.pdist(sampled_coords, 'euclidean')
        mean_vvd = np.mean(pairwise_distances)
    else:
        mean_vvd = 0
    
    # --- Return Results ---
    results = {
        "Vein Density (VD)": vein_density,
        "Vein Thickness (VT)": vein_thickness,
        "Areole Size (AS)": mean_areole_size,
        "Number of Areoles (NA)": num_areoles,
        "Branching Angle (BA)": mean_branching_angle,
        "Vein-to-Vein Distance (VVD)": mean_vvd,
    }
    return results