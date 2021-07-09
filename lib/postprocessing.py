import cv2
import numpy as np


def get_ind(coord, min_coord, max_coord, max_ind, round="up", mem=None):
    # if mem is not None and coord in mem:
    #    return mem[coord]

    p = np.round((coord - min_coord) / (max_coord - min_coord), 4)
    n = p * max_ind

    if round == "up":
        ind = int(np.ceil(n))
    elif (round == "down"):
        ind = int(np.floor(n))
    else:
        ind = n

    return ind


def parse_inds(y1, y2, x1, x2):
    y11, y22, x11, x22 = (int(np.floor(y1)), int(np.ceil(y2)),
                          int(np.floor(x1)), int(np.ceil(x2)))

    thr_mask = np.ones((y22-y11, x22-x11))
    thr_mask[:1, :] *= 1 - (y1 - np.floor(y1))
    thr_mask[-1:, :] *= 1 - (np.ceil(y2) - y2)

    thr_mask[:, :1] *= 1 - (x1 - np.floor(x1))
    thr_mask[:, -1:] *= 1 - (np.ceil(x2) - x2)

    return thr_mask, y11, y22, x11, x22


def postprocess(native_data, bbox, predictions, check=False):
    output = np.zeros_like(predictions)

    predictions = predictions / (predictions.mean() / native_data.mean())
    native_resized = cv2.resize(
        native_data, predictions.shape[::-1], interpolation=cv2.INTER_NEAREST)
    predictions = predictions.copy()
    pos_mask = np.abs(predictions) * 2 > native_resized
    rate = predictions / native_resized
    predictions[pos_mask] = predictions[pos_mask] / rate[pos_mask] / 2

    (min_x, min_y, max_x, max_y) = bbox

    # Get native estimative
    native_size = native_data.shape
    native_solution = np.zeros_like(predictions)
    nb_areas = np.zeros_like(predictions)

    delta_x = max_x - min_x
    deltas_x = np.arange(native_size[1] + 1) * delta_x / (native_size[1])
    coords_x = min_x + deltas_x

    delta_y = max_y - min_y
    deltas_y = np.arange(native_size[0] + 1) * delta_y / (native_size[0])
    coords_y = min_y + deltas_y

    mem_x, mem_y = {}, {}
    max_y_ind, max_x_ind = predictions.shape
    for j in range(native_data.shape[0]):
        y1 = get_ind(coords_y[j], min_y, max_y,
                     max_y_ind, round="down", mem=mem_y)
        y2 = get_ind(coords_y[j+1], min_y, max_y,
                     max_y_ind, round="up", mem=mem_y)
        for i in range(native_data.shape[1]):
            x1 = get_ind(coords_x[i], min_x, max_x,
                         max_x_ind, mem=mem_x, round="down")
            x2 = get_ind(coords_x[i+1], min_x, max_x,
                         max_x_ind, mem=mem_x, round="up")
            native_solution[y1:y2, x1:x2] += native_data[j, i]
            nb_areas[y1:y2, x1:x2] += 1

    native_solution = native_solution / nb_areas

    # Adjust predictions
    improved_predictions = predictions.copy()
    for j in range(native_data.shape[0]):
        y1 = get_ind(coords_y[j], min_y, max_y, max_y_ind, round=False)
        y2 = get_ind(coords_y[j+1], min_y, max_y, max_y_ind, round=False)
        for i in range(native_data.shape[1]):
            x1 = get_ind(coords_x[i], min_x, max_x, max_x_ind, round=False)
            x2 = get_ind(coords_x[i+1], min_x, max_x, max_x_ind, round=False)
            thr_mask, y1, y2, x1, x2 = parse_inds(y1, y2, x1, x2)
            nb_ar = nb_areas[y1:y2, x1:x2]
            nat_ar = native_solution[y1:y2, x1:x2]
            pred_ar = predictions[y1:y2, x1:x2].copy()
            pred_ar[nb_ar == 1] -= (pred_ar *
                                    thr_mask).sum() / (nb_ar == 1).sum()
            pred_ar[nb_ar == 1] -= ((nat_ar * thr_mask).sum() -
                                    native_data[j, i] * (thr_mask).sum()) / (nb_ar == 1).sum()

            improved_predictions[y1:y2, x1:x2] = pred_ar

    improved_solution = native_solution + improved_predictions

    if check:
        # Assert native solution is intact
        for j in range(native_data.shape[0]):
            y1 = get_ind(coords_y[j], min_y, max_y,
                         max_y_ind, round=False, mem=mem_y)
            y2 = get_ind(coords_y[j+1], min_y, max_y,
                         max_y_ind, round=False, mem=mem_y)
            for i in range(native_data.shape[1]):
                x1 = get_ind(coords_x[i], min_x, max_x,
                             max_x_ind, mem=mem_x, round=False)
                x2 = get_ind(coords_x[i+1], min_x, max_x,
                             max_x_ind, mem=mem_x, round=False)
                # assert np.isclose(improved_predictions[y1:y2, x1:x2].mean(), 0)
                thr_mask, y1, y2, x1, x2 = parse_inds(y1, y2, x1, x2)
                sol = (
                    thr_mask * improved_solution[y1:y2, x1:x2]).sum() / thr_mask.sum()

                # if not np.isclose(sol, native_data[j, i]):
                #print(sol, native_data[j, i])
                assert np.isclose(sol, native_data[j, i], rtol=1e-4, atol=1e-4)

    return improved_solution


if __name__ == "__main__":
    # native_data = np.random.random((50, 72))
    # bbox = (6.699999978667811, 42.30000150428628,
    #        13.899999436579254, 47.300000784532074)
    # predictions = np.random.random((565, 560))

    native_data = np.random.random((8, 8))
    bbox = (6.699999978667811, 42.30000150428628,
            13.899999436579254, 47.300000784532074)
    predictions = -np.random.random((66, 61)) * 10

    postprocess(native_data, bbox, predictions, check=True)
