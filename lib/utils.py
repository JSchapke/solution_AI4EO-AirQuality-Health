import copy
import numpy as np


def get_coords(cams_eop):
    grid = cams_eop.data["PM2_5"].squeeze()
    bbox = cams_eop.bbox
    X = bbox.min_x, bbox.max_x
    Y = bbox.min_y, bbox.max_y

    coords = []
    positions = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                x = X[0] * (1 - j / (grid.shape[1] - 1)) + \
                    X[1] * (j / (grid.shape[1]-1))
                y = Y[0] * (1 - i / (grid.shape[0] - 1)) + \
                    Y[1] * (i / (grid.shape[0]-1))
                coords.append((x, y))
                positions.append((i, j))

    return coords, positions


def _get_point(lat, lon, grid, bbox):
    min_x, max_x, min_y, max_y = bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y

    delta_x = max_x - min_x
    deltas_x = np.arange(grid.shape[1]) * delta_x / (grid.shape[1]-1)
    coords_x = min_x + deltas_x
    x_ind = np.argmin(np.abs(coords_x-lon))

    delta_y = max_y - min_y
    deltas_y = np.arange(grid.shape[0]) * delta_y / (grid.shape[0]-1)
    coords_y = min_y + deltas_y[::-1]
    y_ind = np.argmin(np.abs(coords_y-lat))

    return y_ind, x_ind


def get_point(gt_row, eop, key):
    lat, lon = gt_row[["SITE_LATIT", "SITE_LONGI"]]

    grid = eop.data[key[0] if isinstance(key, list) else key][0]
    eop = copy.deepcopy(eop)

    coords = _get_point(lat, lon, grid, eop.bbox)
    y_ind, x_ind = coords

    if isinstance(key, list):
        return {k: eop.data[k][:, y_ind, x_ind] for k in key}, coords
    else:
        data = eop.data[key]
        value = data[:, y_ind, x_ind]
        return value, coords


def get_patch(gt_row, eop, patch_size):
    lat, lon = gt_row[["SITE_LATIT", "SITE_LONGI"]]
    grid = eop.mask_timeless['CORINE'][..., 0]

    coords = _get_point(lat, lon, grid, eop.bbox)
    y_ind, x_ind = coords

    patch = grid[y_ind-patch_size:y_ind+1+patch_size,
                 x_ind-patch_size:x_ind+1+patch_size]
    return patch


def _get_patch(lat, lon, grid, bbox, patch_size):
    coords = _get_point(lat, lon, grid, bbox)
    y_ind, x_ind = coords
    patch = grid[y_ind-patch_size:y_ind+1+patch_size,
                 x_ind-patch_size:x_ind+1+patch_size]
    return patch
