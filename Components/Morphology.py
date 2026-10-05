import numba_morph
import numpy as np
from numba import njit, prange, jit, types
from numba.typed import List, Dict
from heapq import heappush, heappop
from scipy import ndimage
from collections import defaultdict


@njit
def label_h_minima(reconstructed, original, threshold):
    """
    Find connected components of pixels where (reconstructed - original) >= threshold,
    and label them with unique integers (0 for background).

    Parameters
    ----------
    reconstructed : 3D ndarray
        Result of reconstruction by erosion (e.g. from geodesic_reconstruction_by_erosion).
    original : 3D ndarray
        The mask image used in the reconstruction.
    threshold : int or float
        The h value (dynamic) used in reconstruction. Only residual >= threshold are kept.

    Returns
    -------
    labels : 3D ndarray of int32
        Labelled regions (0 = background, positive integers = individual minima).
    """
    Z, Y, X = reconstructed.shape
    labels = np.zeros((Z, Y, X), dtype=np.uint16)

    dirs = ((-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
     (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
     (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
     (0, -1, -1), (0, -1, 0), (0, -1, 1),
     (0, 0, -1), (0, 0, 1), (0, 1, -1),
     (0, 1, 0), (0, 1, 1), (1, -1, -1),
     (1, -1, 0), (1, -1, 1), (1, 0, -1),
     (1, 0, 0), (1, 0, 1), (1, 1, -1),
     (1, 1, 0), (1, 1, 1))

    current_label = 0

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                # Skip already labelled or pixels that do not satisfy the condition
                if labels[z, y, x] != 0:
                    continue
                if reconstructed[z, y, x] - original[z, y, x] < threshold:
                    continue

                # Start a new region
                current_label += 1
                stack = [(z, y, x)]
                labels[z, y, x] = current_label

                # Flood fill (DFS) – uses a list as a stack
                while stack:
                    cz, cy, cx = stack.pop()
                    for dz, dy, dx in dirs:
                        nz = cz + dz
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            if labels[nz, ny, nx] == 0:
                                if reconstructed[nz, ny, nx] - original[nz, ny, nx] >= threshold:
                                    labels[nz, ny, nx] = current_label
                                    stack.append((nz, ny, nx))

    return labels

def __heapify_markers_3d(markers, image):
    """Create a priority queue heap with the markers on it for 3D."""
    stride = np.array(image.strides, dtype=np.uint32) // image.itemsize
    coords = np.argwhere(markers != 0).astype(np.uint32)
    ncoords = coords.shape[0]
    if ncoords > 0:
        pixels = image[markers != 0]
        age = np.arange(ncoords, dtype=np.uint32)
        offset = np.zeros(coords.shape[0], dtype=np.uint32)
        for i in range(image.ndim):
            offset = offset + stride[i] * coords[:, i]
        pq = [tuple(row) for row in np.column_stack((pixels, age, offset, coords))]
        ordering = np.lexsort((age, pixels))
        pq = [pq[i] for i in ordering]
    else:
        pq = np.zeros((0, markers.ndim + 3), int)
    return (pq, ncoords)


@njit(nogil=True)
def _watershed_loop(pq, labels, connect_increments, mask, image, age):
    max_x, max_y, max_z = labels.shape
    total_pixels = image.size if mask is None else np.count_nonzero(mask)
    processed = 0
    print_interval = max(1, total_pixels // 20)  # print every 5%
    print(f"A total of {total_pixels} needs to be processed.")
    while len(pq):
        pix_value, pix_age, _, pix_x, pix_y, pix_z = heappop(pq)
        processed += 1
        pix_label = labels[pix_x, pix_y, pix_z]

        if processed % print_interval == 0:
            progress = processed * 100 // total_pixels
            print(f"Watershed Progress: {progress}% ({processed}/{total_pixels})")

        for dx, dy, dz in connect_increments:
            x, y, z = pix_x + dx, pix_y + dy, pix_z + dz
            if x < 0 or y < 0 or z < 0 or x >= max_x or y >= max_y or z >= max_z:
                continue
            if labels[x, y, z]:
                continue
            if mask is not None and not mask[x, y, z]:
                continue

            labels[x, y, z] = pix_label
            new_pq_item = (np.uint32(image[x, y, z]), np.uint32(age), np.uint32(0), np.uint32(x), np.uint32(y), np.uint32(z))
            heappush(pq, new_pq_item)
            age += 1
    return labels


# The "Slower" watershed taken from scikits-image. Is faster after using Numba.
def marker_controlled_watershed(image, markers, mask=None):
    """Watershed algorithm optimized with Numba for 3D images with 6-connectivity."""
    connect_increments = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ]
    print("Starts watershed flooding...")
    pq, age = __heapify_markers_3d(markers, image)
    return _watershed_loop(pq, markers, connect_increments, mask, image, age)


def inverter(img):
    min_val = img.min()
    max_val = img.max()

    img -= min_val
    np.negative(img, out=img)
    img += max_val

    return img


def remove_small_labels(img, min_size):
    bins = np.bincount(img.ravel())
    for label in bins[bins < min_size]:
        img[img == label] = 0
    return img


@njit(parallel=True)
def pixel_reclaim(touching_map, segmentation, distance_threshold, z_to_xy_ratio=1.01):
    touching_pixels = np.argwhere(touching_map)
    map_size = segmentation.shape
    max_segment_id = segmentation.max()
    segmentation_new = segmentation.copy()

    # Precompute kernel weights based on distance and z_to_xy_ratio
    k_size = 2 * distance_threshold + 1
    kernel = np.zeros((k_size, k_size, k_size), dtype=np.float32)
    center = distance_threshold
    for z_rel in range(k_size):
        dz = z_rel - center
        for y_rel in range(k_size):
            dy = y_rel - center
            for x_rel in range(k_size):
                dx = x_rel - center
                # Calculate weighted distance
                dist = np.sqrt(dx ** 2 + dy ** 2 + (z_to_xy_ratio * dz) ** 2)
                # Weight is inversely proportional to distance
                kernel[z_rel, y_rel, x_rel] = 1.0 / (1.0 + dist)

    for i in prange(touching_pixels.shape[0]):
        z = touching_pixels[i, 0]
        y = touching_pixels[i, 1]
        x = touching_pixels[i, 2]

        z_start = max(z - distance_threshold, 0)
        z_end = min(z + distance_threshold + 1, map_size[0])
        y_start = max(y - distance_threshold, 0)
        y_end = min(y + distance_threshold + 1, map_size[1])
        x_start = max(x - distance_threshold, 0)
        x_end = min(x + distance_threshold + 1, map_size[2])

        # Thread‑local weighted counts
        weighted_counts = np.zeros(max_segment_id + 1, dtype=np.float32)

        for z0 in range(z_start, z_end):
            dz = z0 - z
            k_z = dz + distance_threshold
            for y0 in range(y_start, y_end):
                dy = y0 - y
                k_y = dy + distance_threshold
                for x0 in range(x_start, x_end):
                    dx = x0 - x
                    k_x = dx + distance_threshold
                    segment_id = segmentation[z0, y0, x0]
                    weighted_counts[segment_id] += kernel[k_z, k_y, k_x]

        segment_weights = weighted_counts[1:]
        total_weight = np.sum(segment_weights)
        if total_weight > 0:
            best_segment = np.argmax(segment_weights) + 1
            segmentation_new[z, y, x] = best_segment

    return segmentation_new


@njit(cache=True)
def _score_pairs_numba(seg, contour_dil):
    """
    Scan each positive voxel once. For every *distinct* neighbour label L2 != L,
    bump the counter for the pair (min(L,L2), max(L,L2)).

    The pair key is encoded as an int64:  key = (min << 32) | max.

    Returns
    -------
    keys   : int64 array  (encoded pair keys)
    n_tot  : int64 array  (total boundary voxels per pair)
    n_con  : int64 array  (boundary voxels that fall inside the dilated contour)
    """
    nx, ny, nz = seg.shape

    total = Dict.empty(types.int64, types.int64)
    cont  = Dict.empty(types.int64, types.int64)

    neigh = np.empty(26, dtype=np.int64)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                L = np.int64(seg[x, y, z])
                if L <= 0:
                    continue

                # ---- collect unique neighbour labels (26-connectivity) ----
                n = 0
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= nx:
                        continue
                    for dy in range(-1, 2):
                        yy = y + dy
                        if yy < 0 or yy >= ny:
                            continue
                        for dz in range(-1, 2):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            zz = z + dz
                            if zz < 0 or zz >= nz:
                                continue
                            L2 = np.int64(seg[xx, yy, zz])
                            if L2 <= 0 or L2 == L:
                                continue
                            found = False
                            for k in range(n):
                                if neigh[k] == L2:
                                    found = True
                                    break
                            if not found:
                                neigh[n] = L2
                                n += 1

                if n == 0:
                    continue

                is_c = contour_dil[x, y, z]

                # ---- bump counts for each distinct neighbour label ----
                for k in range(n):
                    L2 = neigh[k]
                    if L < L2:
                        key = (L << 32) | L2
                    else:
                        key = (L2 << 32) | L

                    total[key] = total.get(key, 0) + 1
                    if is_c:
                        cont[key] = cont.get(key, 0) + 1

    m = len(total)
    keys = np.empty(m, dtype=np.int64)
    nt   = np.empty(m, dtype=np.int64)
    nc   = np.empty(m, dtype=np.int64)
    i = 0
    for k in total.keys():
        keys[i] = k
        nt[i]   = total[k]
        nc[i]   = cont.get(k, 0)
        i += 1
    return keys, nt, nc


@njit(cache=True, parallel=True)
def _apply_label_map_inplace(seg, label_map):
    """Replace seg[x,y,z] -> label_map[seg[x,y,z]] in-place."""
    nx, ny, nz = seg.shape
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                v = seg[x, y, z]
                if v > 0:
                    seg[x, y, z] = label_map[v]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def rag_merge_by_contour(segmentation,
                         contour_map,
                         merge_threshold=0.10,
                         contour_dilation=1,
                         max_iterations=20,
                         verbose=True):
    """
    Iteratively merge labels whose shared boundary has low contour support.

    Modifies `segmentation` in place

    Parameters
    ----------
    segmentation : np.ndarray (integer)
        Labeled volume. 0 = background. Modified **in place**.
    contour_map : np.ndarray (bool or 0/1)
        Binary map, True where a contour / boundary is expected.
    merge_threshold : float
        Merge a pair when (contour-supported boundary voxels / total boundary
        voxels) is <= this value.
    contour_dilation : int
        Radius (26-connectivity) to dilate the contour map.
    max_iterations : int
        Safety cap on the merge loop.
    verbose : bool
    """
    seg = segmentation          # in-place working buffer
    dtype = seg.dtype

    # -- one-time dilation of the contour map ------------------------------
    contour = contour_map
    if contour.dtype != np.bool_:
        contour = contour.astype(bool)

    if contour_dilation > 0:
        struct = ndimage.generate_binary_structure(3, 3)
        contour_dil = numba_morph.dilation(contour, footprint=struct, iterations=int(contour_dilation))
    else:
        contour_dil = contour

    # -- iterate ----------------------------------------------------------
    for it in range(max_iterations):
        keys, n_total, n_contour = _score_pairs_numba(seg, contour_dil)
        n_edges = keys.shape[0]

        if n_edges == 0:
            if verbose:
                print(f"[rag] iter {it}: no adjacent labels, done.")
            break

        # n_total is always > 0 by construction
        ratios = n_contour / n_total
        candidate = ratios <= merge_threshold

        if verbose:
            print(f"[rag] iter {it}: {n_edges} edge(s), "
                  f"{int(candidate.sum())} below threshold ({merge_threshold}).")

        if not candidate.any():
            break

        # ---------- union-find over candidate merges ---------------------
        parent = {}

        def find(x):
            r = x
            while parent.get(r, r) != r:
                r = parent[r]
            # path compression
            while parent.get(x, x) != x:
                nxt = parent[x]
                parent[x] = r
                x = nxt
            return r

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n_edges):
            if candidate[i]:
                k = int(keys[i])
                union(k >> 32, k & 0xFFFFFFFF)

        # ---------- build old -> new label map ---------------------------
        unique_labels = np.unique(seg)
        unique_labels = unique_labels[unique_labels > 0]

        old_to_new = {}
        root_to_new = {}
        next_new = 1
        for lbl in unique_labels:
            l = int(lbl)
            r = find(l)
            if r not in root_to_new:
                root_to_new[r] = next_new
                next_new += 1
            old_to_new[l] = root_to_new[r]

        if next_new == len(unique_labels) + 1 and \
                all(l == old_to_new[l] for l in old_to_new):
            # nothing was merged
            if verbose:
                print(f"[rag] iter {it}: no labels changed, stopping.")
            break

        # dense lookup table indexed by old label (labels are small ints)
        max_lbl = int(unique_labels.max())
        label_map = np.zeros(max_lbl + 1, dtype=dtype)
        for old, new in old_to_new.items():
            label_map[old] = new

        # ---------- in-place relabel -------------------------------------
        _apply_label_map_inplace(seg, label_map)
