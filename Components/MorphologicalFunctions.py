import numpy as np
from numba import njit, prange
from heapq import heappush, heappop


def geodesicreconstructionbyerosion3d(mask, dynamic):
    result = mask + dynamic
    mod_if = True
    print("Geodesic Reconstructing...")
    while mod_if:
        mod_if = False
        result, mod_if = _forward_scan_c6(mask, result, mod_if)
        result, mod_if = _backward_scan_c6(mask, result, mod_if)
    return result


@njit
def _forward_scan_c6(mask, result, mod_if):
    for z in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[2]):
                current_value = result[z, y, x]
                min_value = current_value

                if x > 0:
                    min_value = min(min_value, result[z, y, x-1])
                if y > 0:
                    min_value = min(min_value, result[z, y-1, x])
                if z > 0:
                    min_value = min(min_value, result[z-1, y, x])

                min_value = max(min_value, mask[z, y, x])
                if min_value < current_value:
                    result[z, y, x] = min_value
                    mod_if = True
    return result, mod_if


@njit
def _backward_scan_c6(mask, result, mod_if):
    for z in range(mask.shape[0] - 1, -1, -1):
        for y in range(mask.shape[1] - 1, -1, -1):
            for x in range(mask.shape[2] - 1, -1, -1):
                current_value = result[z, y, x]
                min_value = current_value

                if x < mask.shape[2] - 1:
                    min_value = min(min_value, result[z, y, x+1])
                if y < mask.shape[1] - 1:
                    min_value = min(min_value, result[z, y+1, x])
                if z < mask.shape[0] - 1:
                    min_value = min(min_value, result[z+1, y, x])

                min_value = max(min_value, mask[z, y, x])
                if min_value < current_value:
                    result[z, y, x] = min_value
                    mod_if = True
    return result, mod_if


def chamferdistancetransform3duint16(img):
    result = np.where(img > 0, np.uint16(np.iinfo(np.uint16).max), np.uint16(0))
    result = _forward_scan_cham_c6(img, result)
    result = _backward_scan_cham_c6(img, result)
    return result

# Borgefors 3D chamfer weights (26‑neighbourhood)
# Define the Borgefors weights and offsets
offsets = (
    (1, 0, 0, 3),
    (0, 1, 0, 3),
    (0, 0, 1, 3),
    (-1, 0, 0, 3),
    (0, -1, 0, 3),
    (0, 0, -1, 3),
    (1, 1, 0, 4),
    (1, -1, 0, 4),
    (-1, 1, 0, 4),
    (-1, -1, 0, 4),
    (1, 0, 1, 4),
    (1, 0, -1, 4),
    (-1, 0, 1, 4),
    (-1, 0, -1, 4),
    (0, 1, 1, 4),
    (0, 1, -1, 4),
    (0, -1, 1, 4),
    (0, -1, -1, 4),
    (1, 1, 1, 5),
    (1, 1, -1, 5),
    (1, -1, 1, 5),
    (1, -1, -1, 5),
    (-1, -1, 1, 5),
    (-1, 1, 1, 5),
    (-1, 1, -1, 5),
    (-1, -1, -1, 5))

@njit(nogil=True)
def _forward_scan_cham_c6(img, result):
    for z in range(img.shape[0]):
        for y in range(img.shape[1]):
            for x in range(img.shape[2]):
                if img[z, y, x] == 0:
                    continue

                current_value = result[z, y, x]
                new_value = np.iinfo(np.uint16).max

                # Iterate over the offsets
                for dx, dy, dz, weight in offsets:
                    x2 = x + dx
                    y2 = y + dy
                    z2 = z + dz

                    # Check if the neighbor is within bounds
                    if 0 <= x2 < img.shape[2] and 0 <= y2 < img.shape[1] and 0 <= z2 < img.shape[0]:
                        neighbor_value = result[z2, y2, x2] + weight
                        new_value = min(new_value, neighbor_value)

                # Update the current voxel if a smaller value was found
                if new_value < current_value:
                    result[z, y, x] = new_value
    return result


@njit(nogil=True)
def _backward_scan_cham_c6(img, result):
    for z in range(img.shape[0] - 1, -1, -1):
        for y in range(img.shape[1] - 1, -1, -1):
            for x in range(img.shape[2] - 1, -1, -1):
                if img[z, y, x] == 0:
                    continue

                current_value = result[z, y, x]
                new_value = np.iinfo(np.uint16).max

                # Iterate over the offsets
                for dx, dy, dz, weight in offsets:
                    x2 = x + dx
                    y2 = y + dy
                    z2 = z + dz

                    # Check if the neighbor is within bounds
                    if 0 <= x2 < img.shape[2] and 0 <= y2 < img.shape[1] and 0 <= z2 < img.shape[0]:
                        neighbor_value = result[z2, y2, x2] + weight
                        new_value = min(new_value, neighbor_value)

                # Update the current voxel if a smaller value was found
                if new_value < current_value:
                    result[z, y, x] = new_value
    return result


@njit(parallel=True, nogil=True)
def chamfer_distance_transform_parallel(binary_mask, num_core=16):
    """
    Parallel chamfer distance transform with early stopping.
    """
    Z, Y, X = binary_mask.shape
    num_bands = num_core
    chunk_size = (Z // num_bands) + 1
    max_val = np.iinfo(np.uint16).max

    result = np.where(binary_mask > 0, np.uint16(max_val), np.uint16(0))
    print(f"Need maximum {Z} Iteration for Chamfer Distance Transform")

    # 3D chamfer weights: face=3, edge=4, corner=5
    causal = [
        (-1, -1, -1, 5), (-1, -1, 0, 4), (-1, -1, 1, 5),
        (-1, 0, -1, 4), (-1, 0, 0, 3), (-1, 0, 1, 4),
        (-1, 1, -1, 5), (-1, 1, 0, 4), (-1, 1, 1, 5),
        (0, -1, -1, 4), (0, -1, 0, 3), (0, -1, 1, 4),
        (0, 0, -1, 3),
    ]
    anti_causal = [
        (1, -1, -1, 5), (1, -1, 0, 4), (1, -1, 1, 5),
        (1, 0, -1, 4), (1, 0, 0, 3), (1, 0, 1, 4),
        (1, 1, -1, 5), (1, 1, 0, 4), (1, 1, 1, 5),
        (0, 1, -1, 4), (0, 1, 0, 3), (0, 1, 1, 4),
        (0, 0, 1, 3),
    ]

    # We'll iterate at most Z times (worst‑case path length)
    for it in range(Z):
        print(f"Iteration {it} for Chamfer Distance Transform")
        changed = np.zeros(num_bands, dtype=np.bool_)   # reset for this iteration

        # ---- Forward pass (parallel over bands) ----
        for band in prange(num_bands):
            start_z = band * chunk_size
            end_z = min(start_z + chunk_size, Z)
            for z in range(start_z, end_z):
                for y in range(Y):
                    for x in range(X):
                        if binary_mask[z, y, x] == 0:
                            continue
                        cur = result[z, y, x]
                        new_val = cur
                        for dz, dy, dx, w in causal:
                            nz = z + dz
                            ny = y + dy
                            nx = x + dx
                            if (0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X):
                                neigh = result[nz, ny, nx]
                                if neigh != max_val:
                                    cand = neigh + w
                                    if cand < new_val:
                                        new_val = cand
                        if new_val < cur:
                            result[z, y, x] = new_val
                            changed[band] = True   # record that this band was updated

        # ---- Backward pass (parallel over bands) ----
        for band in prange(num_bands):
            start_z = band * chunk_size
            end_z = min(start_z + chunk_size, Z)
            for z in range(end_z - 1, start_z - 1, -1):
                for y in range(Y - 1, -1, -1):
                    for x in range(X - 1, -1, -1):
                        if binary_mask[z, y, x] == 0:
                            continue
                        cur = result[z, y, x]
                        new_val = cur
                        for dz, dy, dx, w in anti_causal:
                            nz = z + dz
                            ny = y + dy
                            nx = x + dx
                            if (0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X):
                                neigh = result[nz, ny, nx]
                                if neigh != max_val:
                                    cand = neigh + w
                                    if cand < new_val:
                                        new_val = cand
                        if new_val < cur:
                            result[z, y, x] = new_val
                            changed[band] = True   # record that this band was updated

        # After both passes, check if any change occurred anywhere
        if not np.any(changed):
            print("No more change detected! Stopping Early.")
            break   # converged

    return result

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
    while len(pq):
        pix_value, pix_age, _, pix_x, pix_y, pix_z = heappop(pq)
        pix_label = labels[pix_x, pix_y, pix_z]

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
def watershed_3d(image, markers, mask=None):
    """Watershed algorithm optimized with Numba for 3D images with 6-connectivity."""
    connect_increments = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ]
    print("Starts watershed flooding...")
    pq, age = __heapify_markers_3d(markers, image)
    print('Watersheding...')
    return _watershed_loop(pq, markers, connect_increments, mask, image, age)


def inverter(img):
    min_val = img.min()
    max_val = img.max()

    img -= min_val
    np.negative(img, out=img)
    img += max_val

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