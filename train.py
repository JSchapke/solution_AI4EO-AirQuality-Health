# Built-in modules
import argparse
import os
import gzip
import random
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm
#from shapely import wkt

from lib.utils import _get_point
from configs.utils import load_config
from lib.train_utils import get_data, load_pickle, save_pickle
from algorithms.ensemble import Ensemble


def preprocess(config, update=False):
    if not os.path.isfile(config["cache_path"]) or update:
        data, gts, target_size = get_data(
            indir, outdir, label, aoi, config["feature_keys"], update=update)
        save_pickle((data, gts, target_size), config["cache_path"])
    else:
        (data, gts, target_size) = load_pickle(config["cache_path"])

    # Dataset
    dataset = {"X": [],
               "Y": [],
               "gt": [],
               "native_Y": [],
               "target_Y": [],
               "lat": [],
               "lon": [],
               "coords": [],
               "date": []}
    grid = np.zeros(target_size)
    bbox = data['data'][0]["s5p" if label == "NO2" else "cams"].bbox
    for date, lat, lon, gt in tqdm.tqdm(gts[["Date", "SITE_LATIT", "SITE_LONGI", "AirQuality"]].values):
        y_ind, x_ind = _get_point(lat, lon, grid, bbox)
        try:
            day_ind = data["date"].index(int(date))

            ind = data['date'].index(date)
            native = data['feat_dicts'][ind]['no2_native' if label ==
                                             'NO2' else 'pm25_native']
            target = data['feat_dicts'][ind]['no2_target' if label ==
                                             'NO2' else 'pm25_target']

            if np.isnan(native[y_ind, x_ind]):
                continue

            dataset["X"].append(data['feats'][day_ind][y_ind, x_ind])
            dataset["native_Y"].append(native[y_ind, x_ind])
            dataset["target_Y"].append(target[y_ind, x_ind])
            dataset["gt"].append(gt)
            dataset["lat"].append(lat)
            dataset["lon"].append(lon)
            dataset["coords"].append((lat, lon))
            dataset["date"].append(date)

        except:
            pass
            #print("Error on date:", date)

    dataset = {k: np.array(v) for k, v in dataset.items()}
    # -------------------

    gt, native_Y = dataset["gt"], dataset["native_Y"]
    # Filter any very off GT

    if label == "NO2":
        dataset["Y"] = gt / (6.02214 * 1e4 * 1.9125) - native_Y
        v1 = gt
        v2 = native_Y * 6.02214 * 1e4 * 1.9125
        m = np.min([v2, v1], axis=0)
        r = (v1 / v2)
        if "Africa" in aoi:
            mask = (r < 3.5) & (r > 0.7)
        elif "California" == aoi:
            mask = (r < 4.5) & (r > 0.7)
        elif "Italy" == aoi:
            mask = (r < 2.5) & (r > 0.85)

    else:
        dataset["Y"] = gt - native_Y
        v1 = gt
        v2 = native_Y
        if "Africa" in aoi:
            m = np.min([v2, v1], axis=0)
            mask = ~((abs(v2 - v1) > (m * 3)) & ((v2 > 1) | (v1 > 1)))
        elif aoi == "Italy":
            mask = ~((abs(v2 - v1) > v2 * 0.9) & ((v2 > 5) | (v1 > 5)))
        elif aoi == "California":
            mask = ~((abs(v2 - v1) > v2 * 0.535) & ((v2 > 5) | (v1 > 5)))

    print('Rate of kept samples from filter:', mask.mean())
    print('Correlation between native measure and ground truth',
          np.corrcoef(gt[mask], native_Y[mask]))

    X = dataset["X"][mask]
    Y = dataset["Y"][mask]
    coords = dataset["coords"][mask]
    return X, Y, coords, data


def main(config, update=False):
    X, Y, coords, data = preprocess(config, update=update)

    ensemble = Ensemble(config, indir, outdir)
    print("models:", ensemble.models)

    mu = np.nanmean(data["feats"], axis=(0, 1, 2))
    sigma = np.nanstd(data["feats"], axis=(0, 1, 2))

    ensemble.train(X, Y, coords=coords, mu=mu, sigma=sigma)
    ensemble.save()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--update', action="store_true")
    args = parser.parse_args()

    random.seed(0)

    config = load_config(args.config)
    aoi = config["aoi"]
    label = config["label"]
    indir = Path("./data/train") / config["aoi"]
    outdir = Path("./wdir") / config["aoi"]

    main(config, args.update)
