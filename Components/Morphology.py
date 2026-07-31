import numpy as np
from numba import njit, prange, jit, types
from numba.typed import List
from heapq import heappush, heappop


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