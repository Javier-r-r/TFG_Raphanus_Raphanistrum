"""Metric computations for vein networks detected in binary masks.

This module provides functions to compute normalized and raw measurements
from vein segmentation masks: density, thickness (centerline-based),
areole size and count, branching angles, vein-to-vein distances and
main vein counts. Functions accept NumPy arrays and return plain Python
types or NumPy values suitable for writing to CSV or further analysis.
"""
import numpy as np
import networkx as nx

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from math import atan2, degrees
from petals_generator import generate_petal_mask_from_rgb


def compute_normalized_metrics(mask, petal_mask=None, img_rgb=None, reference_resolution=1000):
    """Compute metrics scaled to a common reference resolution.

    The function delegates metric extraction to `compute_vein_metrics` and
    then scales size-dependent metrics (thickness, areole size, vein-to-vein
    distance) so values are comparable across images with different
    resolutions. `reference_resolution` denotes the diagonal in pixels used
    as the target normalization basis.

    Args:
        mask (np.ndarray): Binary vein mask (0/255 or boolean).
        petal_mask (np.ndarray, optional): Binary mask of the petal area.
        img_rgb (np.ndarray, optional): RGB image used to infer petal mask.
        reference_resolution (float): Diagonal (pixels) to scale to.

    Returns:
        dict: Same keys as `compute_vein_metrics` with selected metrics scaled.
    """
    results = compute_vein_metrics(mask, petal_mask, img_rgb)

    height, width = mask.shape
    image_diagonal = np.sqrt(height**2 + width**2)
    scale_factor = reference_resolution / image_diagonal

    results["Vein Thickness (VT)"] *= scale_factor
    results["Areole Size (AS)"] *= scale_factor
    results["Vein-to-Vein Distance (VVD)"] *= scale_factor

    return results


def compute_vein_thickness_centerline(binary_mask: np.ndarray):
    """Estimate vein thickness from skeleton centerline distances.

    The routine skeletonizes the binary mask and builds a graph where
    skeleton pixels are nodes. Only intermediate nodes (degree == 2)
    are considered to avoid endpoints and branch junctions. The distance
    transform evaluated at these centerline points provides a local radius
    estimate; thickness is radius * 2 and the function returns the mean
    thickness across sampled centerline points.

    Args:
        binary_mask (np.ndarray): Boolean or 0/255 mask of veins.

    Returns:
        float: Mean estimated thickness in pixels (0.0 if not computable).
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0

    sk = skeletonize(binary_mask)

    coords = np.column_stack(np.where(sk))
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dy, dx) != (0, 0) and (y + dy, x + dx) in G.nodes:
                    G.add_edge((y, x), (y + dy, x + dx))

    mid_points = [n for n in G.nodes if G.degree[n] == 2]
    if not mid_points:
        return np.array([]), 0.0, 0.0, 0.0

    dist = distance_transform_edt(binary_mask)
    radii = np.array([dist[y, x] for y, x in mid_points])
    local_thickness = radii * 2.0

    mean_t = float(np.mean(local_thickness)) if local_thickness.size else 0.0

    return mean_t


def compute_adjacent_vein_distances(skeleton, binary_mask, G, n_samples=200):
    """Compute distances between adjacent veins.

    This function samples skeleton points that belong to simple segments
    (degree == 2) and searches perpendicular to the local tangent to find
    the nearest adjacent foreground after leaving the original vein. When
    perpendicular sampling yields no result, a fallback using cKDTree on
    labeled skeleton components is used to compute pairwise minima.

    Args:
        skeleton (np.ndarray): Skeletonized binary mask.
        binary_mask (np.ndarray): Binary mask used to detect foreground.
        G (networkx.Graph): Graph built from the skeleton pixels.
        n_samples (int): Maximum number of skeleton points to sample.

    Returns:
        list[float]: Distances (in pixels) between veins; empty list if none.
    """
    coords = np.array([n for n in G.nodes if G.degree[n] == 2])
    if len(coords) == 0:
        return []

    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]

    labeled = label(binary_mask, connectivity=1)

    distances = []
    for y, x in coords:
        neighbors = list(G.neighbors((y, x)))
        if len(neighbors) != 2:
            continue
        (y1, x1), (y2, x2) = neighbors
        dy = y2 - y1
        dx = x2 - x1
        norm = np.hypot(dy, dx)
        if norm == 0:
            continue
        perp_y1 = int(round(-dx / norm))
        perp_x1 = int(round(dy / norm))
        perp_y2 = -perp_y1
        perp_x2 = -perp_x1

        current_label = labeled[y, x] if (0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]) else 0

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
                    found_background = True
                    continue
                if not found_background:
                    continue
                min_dist = d if (min_dist is None or d < min_dist) else min_dist
                break
        if min_dist is not None:
            distances.append(min_dist)

    if not distances:
        sk_mask = (skeleton > 0)
        ys, xs = np.where(sk_mask)
        if ys.size == 0:
            return []
        labels_at_points = labeled[ys, xs]
        unique_labels = np.unique(labels_at_points)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size < 2:
            return []

        points_by_label = {}
        for lab in unique_labels:
            mask_lab = (labels_at_points == lab)
            pts = np.column_stack((ys[mask_lab], xs[mask_lab]))
            if pts.size:
                points_by_label[lab] = pts

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
                dists, _ = tree_i.query(pts_j, k=1)
                min_pair_dist = float(np.min(dists)) if dists.size else None
                if min_pair_dist is not None:
                    distances.append(min_pair_dist)
    return distances


def count_main_veins(mask, min_length=30):
    """Count main veins in a binary mask by following skeleton branches.

    The function skeletonizes the mask, builds a graph of skeleton pixels
    and follows branches starting from endpoints. Branches whose length
    exceeds `min_length` are counted as main veins.

    Args:
        mask (np.ndarray): Binary vein mask.
        min_length (int): Minimum branch length (pixels) to count.

    Returns:
        int: Number of main veins detected.
    """
    skeleton = skeletonize(mask > 0)

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

    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    visited = set()
    vein_count = 0

    for ep in endpoints:
        if ep in visited:
            continue
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

        if len(path) >= min_length:
            vein_count += 1

    return vein_count


def compute_vein_metrics(mask: np.ndarray, petal_mask: np.ndarray = None, img_rgb: np.ndarray = None) -> dict:
    """Compute a set of vein network metrics from a binary mask.

    The returned dictionary contains:
      - Vein Density (VD)
      - Vein Thickness (VT)
      - Areole Size (AS)
      - Number of Areoles (NA)
      - Branching Angle (BA)
      - Vein-to-Vein Distance (VVD)
      - Main Veins (MV)

    If `petal_mask` is not provided and `img_rgb` is available, a petal
    mask is inferred using `generate_petal_mask_from_rgb` and combined with
    the vein mask for area-based metrics.

    Args:
        mask (np.ndarray): Binary mask of veins (0/255 or boolean).
        petal_mask (np.ndarray, optional): Binary petal mask.
        img_rgb (np.ndarray, optional): RGB image used to infer petal mask.

    Returns:
        dict: Computed metrics keyed by human-friendly names.
    """
    binary_mask = mask > 0 if mask.dtype != bool else mask
    skeleton = skeletonize(binary_mask, method='zhang')

    vein_area = np.sum(binary_mask)
    if petal_mask is None and img_rgb is not None:
        petal_mask = generate_petal_mask_from_rgb(img_rgb)
        petal_mask = np.clip(petal_mask + (binary_mask.astype(np.uint8)), 0, 1)
    petal_area = np.sum(petal_mask > 0) if petal_mask is not None else binary_mask.size
    vein_density = vein_area / petal_area if petal_area > 0 else 0

    vt_mean = float(compute_vein_thickness_centerline(binary_mask))

    inverted_mask = np.logical_not(binary_mask)
    labeled_areoles = label(inverted_mask, connectivity=1)
    regions = regionprops(labeled_areoles)
    min_areole_size = 10

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
        if valid_regions:
            for r in valid_regions:
                if r is largest_region:
                    continue
                equivalent_diameter = 2 * np.sqrt(r.area / np.pi)
                areole_diameters.append(equivalent_diameter)
    num_areoles = len(areole_diameters)
    mean_areole_size = np.mean(areole_diameters) if areole_diameters else 0

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

    vvd_distances = compute_adjacent_vein_distances(skeleton, binary_mask, G)
    mean_vvd = np.mean(vvd_distances) if vvd_distances else 0

    main_veins = count_main_veins(binary_mask, min_length=5)

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
import numpy as np
import networkx as nx

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from math import atan2, degrees
from petals_generator import generate_petal_mask_from_rgb


def compute_normalized_metrics(mask, petal_mask=None, img_rgb=None, reference_resolution=1000):
    """Compute metrics scaled to a common reference resolution.

    The function delegates metric extraction to `compute_vein_metrics` and
    then scales size-dependent metrics (thickness, areole size, vein-to-vein
    distance) so values are comparable across images with different
    resolutions. `reference_resolution` denotes the diagonal in pixels used
    as the target normalization basis.

    Args:
        mask (np.ndarray): Binary vein mask (0/255 or boolean).
        petal_mask (np.ndarray, optional): Binary mask of the petal area.
        img_rgb (np.ndarray, optional): RGB image used to infer petal mask.
        reference_resolution (float): Diagonal (pixels) to scale to.

    Returns:
        dict: Same keys as `compute_vein_metrics` with selected metrics scaled.
    """
    results = compute_vein_metrics(mask, petal_mask, img_rgb)

    height, width = mask.shape
    image_diagonal = np.sqrt(height**2 + width**2)
    scale_factor = reference_resolution / image_diagonal

    results["Vein Thickness (VT)"] *= scale_factor
    results["Areole Size (AS)"] *= scale_factor
    results["Vein-to-Vein Distance (VVD)"] *= scale_factor

    return results


def compute_vein_thickness_centerline(binary_mask: np.ndarray):
    """Estimate vein thickness from skeleton centerline distances.

    The routine skeletonizes the binary mask and builds a graph where
    skeleton pixels are nodes. Only intermediate nodes (degree == 2)
    are considered to avoid endpoints and branch junctions. The distance
    transform evaluated at these centerline points provides a local radius
    estimate; thickness is radius * 2 and the function returns the mean
    thickness across sampled centerline points.

    Args:
        binary_mask (np.ndarray): Boolean or 0/255 mask of veins.

    Returns:
        float: Mean estimated thickness in pixels (0.0 if not computable).
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0

    sk = skeletonize(binary_mask)

    coords = np.column_stack(np.where(sk))
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dy, dx) != (0, 0) and (y + dy, x + dx) in G.nodes:
                    G.add_edge((y, x), (y + dy, x + dx))

    mid_points = [n for n in G.nodes if G.degree[n] == 2]
    if not mid_points:
        return np.array([]), 0.0, 0.0, 0.0

    dist = distance_transform_edt(binary_mask)
    radii = np.array([dist[y, x] for y, x in mid_points])
    local_thickness = radii * 2.0

    mean_t = float(np.mean(local_thickness)) if local_thickness.size else 0.0

    return mean_t


def compute_adjacent_vein_distances(skeleton, binary_mask, G, n_samples=200):
    """Compute distances between adjacent veins.

    This function samples skeleton points that belong to simple segments
    (degree == 2) and searches perpendicular to the local tangent to find
    the nearest adjacent foreground after leaving the original vein. When
    perpendicular sampling yields no result, a fallback using cKDTree on
    labeled skeleton components is used to compute pairwise minima.

    Args:
        skeleton (np.ndarray): Skeletonized binary mask.
        binary_mask (np.ndarray): Binary mask used to detect foreground.
        G (networkx.Graph): Graph built from the skeleton pixels.
        n_samples (int): Maximum number of skeleton points to sample.

    Returns:
        list[float]: Distances (in pixels) between veins; empty list if none.
    """
    coords = np.array([n for n in G.nodes if G.degree[n] == 2])
    if len(coords) == 0:
        return []

    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]

    labeled = label(binary_mask, connectivity=1)

    distances = []
    for y, x in coords:
        neighbors = list(G.neighbors((y, x)))
        if len(neighbors) != 2:
            continue
        (y1, x1), (y2, x2) = neighbors
        dy = y2 - y1
        dx = x2 - x1
        norm = np.hypot(dy, dx)
        if norm == 0:
            continue
        perp_y1 = int(round(-dx / norm))
        perp_x1 = int(round(dy / norm))
        perp_y2 = -perp_y1
        perp_x2 = -perp_x1

        current_label = labeled[y, x] if (0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]) else 0

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
                    found_background = True
                    continue
                if not found_background:
                    continue
                min_dist = d if (min_dist is None or d < min_dist) else min_dist
                break
        if min_dist is not None:
            distances.append(min_dist)

    if not distances:
        sk_mask = (skeleton > 0)
        ys, xs = np.where(sk_mask)
        if ys.size == 0:
            return []
        labels_at_points = labeled[ys, xs]
        unique_labels = np.unique(labels_at_points)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size < 2:
            return []

        points_by_label = {}
        for lab in unique_labels:
            mask_lab = (labels_at_points == lab)
            pts = np.column_stack((ys[mask_lab], xs[mask_lab]))
            if pts.size:
                points_by_label[lab] = pts

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
                dists, _ = tree_i.query(pts_j, k=1)
                min_pair_dist = float(np.min(dists)) if dists.size else None
                if min_pair_dist is not None:
                    distances.append(min_pair_dist)
    return distances


def count_main_veins(mask, min_length=30):
    """Count main veins in a binary mask by following skeleton branches.

    The function skeletonizes the mask, builds a graph of skeleton pixels
    and follows branches starting from endpoints. Branches whose length
    exceeds `min_length` are counted as main veins.

    Args:
        mask (np.ndarray): Binary vein mask.
        min_length (int): Minimum branch length (pixels) to count.

    Returns:
        int: Number of main veins detected.
    """
    skeleton = skeletonize(mask > 0)

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

    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    visited = set()
    vein_count = 0

    for ep in endpoints:
        if ep in visited:
            continue
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

        if len(path) >= min_length:
            vein_count += 1

    return vein_count


def compute_vein_metrics(mask: np.ndarray, petal_mask: np.ndarray = None, img_rgb: np.ndarray = None) -> dict:
    """Compute a set of vein network metrics from a binary mask.

    The returned dictionary contains:
      - Vein Density (VD)
      - Vein Thickness (VT)
      - Areole Size (AS)
      - Number of Areoles (NA)
      - Branching Angle (BA)
      - Vein-to-Vein Distance (VVD)
      - Main Veins (MV)

    If `petal_mask` is not provided and `img_rgb` is available, a petal
    mask is inferred using `generate_petal_mask_from_rgb` and combined with
    the vein mask for area-based metrics.

    Args:
        mask (np.ndarray): Binary mask of veins (0/255 or boolean).
        petal_mask (np.ndarray, optional): Binary petal mask.
        img_rgb (np.ndarray, optional): RGB image used to infer petal mask.

    Returns:
        dict: Computed metrics keyed by human-friendly names.
    """
    binary_mask = mask > 0 if mask.dtype != bool else mask
    skeleton = skeletonize(binary_mask, method='zhang')

    vein_area = np.sum(binary_mask)
    if petal_mask is None and img_rgb is not None:
        petal_mask = generate_petal_mask_from_rgb(img_rgb)
        petal_mask = np.clip(petal_mask + (binary_mask.astype(np.uint8)), 0, 1)
    petal_area = np.sum(petal_mask > 0) if petal_mask is not None else binary_mask.size
    vein_density = vein_area / petal_area if petal_area > 0 else 0

    vt_mean = float(compute_vein_thickness_centerline(binary_mask))

    inverted_mask = np.logical_not(binary_mask)
    labeled_areoles = label(inverted_mask, connectivity=1)
    regions = regionprops(labeled_areoles)
    min_areole_size = 10

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
        if valid_regions:
            for r in valid_regions:
                if r is largest_region:
                    continue
                equivalent_diameter = 2 * np.sqrt(r.area / np.pi)
                areole_diameters.append(equivalent_diameter)
    num_areoles = len(areole_diameters)
    mean_areole_size = np.mean(areole_diameters) if areole_diameters else 0

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

    vvd_distances = compute_adjacent_vein_distances(skeleton, binary_mask, G)
    mean_vvd = np.mean(vvd_distances) if vvd_distances else 0

    main_veins = count_main_veins(binary_mask, min_length=5)

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
