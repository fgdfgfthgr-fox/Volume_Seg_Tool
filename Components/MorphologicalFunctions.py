import numpy as np
from numba import njit
from heapq import heappush, heappop


def geodesicreconstructionbyerosion3d(marker, mask):
    result = np.maximum(marker, mask)
    mod_if = True
    print("Geodesic Reconstructing...")
    while mod_if:
        mod_if = False
        result, mod_if = _forward_scan_c6(marker, mask, result, mod_if)
        result, mod_if = _backward_scan_c6(marker, mask, result, mod_if)
    return result


@njit
def _forward_scan_c6(marker, mask, result, mod_if):
    for z in range(marker.shape[0]):
        for y in range(marker.shape[1]):
            for x in range(marker.shape[2]):
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
def _backward_scan_c6(marker, mask, result, mod_if):
    for z in range(marker.shape[0] - 1, -1, -1):
        for y in range(marker.shape[1] - 1, -1, -1):
            for x in range(marker.shape[2] - 1, -1, -1):
                current_value = result[z, y, x]
                min_value = current_value

                if x < marker.shape[2] - 1:
                    min_value = min(min_value, result[z, y, x+1])
                if y < marker.shape[1] - 1:
                    min_value = min(min_value, result[z, y+1, x])
                if z < marker.shape[0] - 1:
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


@njit
def _forward_scan_cham_c6(img, result):
    # Define the Borgefors weights and offsets
    offsets = [
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
        (-1, -1, -1, 5),
    ]

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


@njit
def _backward_scan_cham_c6(img, result):
    # Define the Borgefors weights and offsets
    offsets = [
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
        (-1, -1, -1, 5),
    ]

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


@njit
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
    pq, age = __heapify_markers_3d(markers, image)
    print('Watersheding...')
    return _watershed_loop(pq, markers, connect_increments, mask, image, age)


def inverter(img):
    min = img.min()
    max = img.max()
    img = max - (img - min)
    return img


@njit
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

    for pixel_idx in touching_pixels:
        z, y, x = pixel_idx[0], pixel_idx[1], pixel_idx[2]
        # Define local region boundaries
        z_start = max(z - distance_threshold, 0)
        z_end = min(z + distance_threshold + 1, map_size[0])
        y_start = max(y - distance_threshold, 0)
        y_end = min(y + distance_threshold + 1, map_size[1])
        x_start = max(x - distance_threshold, 0)
        x_end = min(x + distance_threshold + 1, map_size[2])

        # Initialize weighted counts for segments
        weighted_counts = np.zeros(max_segment_id + 1, dtype=np.float32)

        # Iterate over the local region
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
                    # Add kernel weight to the segment's count
                    weighted_counts[segment_id] += kernel[k_z, k_y, k_x]

        # Ignore background (segment ID 0)
        segment_weights = weighted_counts[1:]
        total_weight = np.sum(segment_weights)
        if total_weight > 0:
            best_segment = np.argmax(segment_weights) + 1
            segmentation_new[z, y, x] = best_segment

    return segmentation_new