import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
# Imports from eo-learn and sentinelhub-py
from eolearn.core import EOPatch, EOTask, LinearWorkflow, FeatureType
from sentinelhub import bbox_to_dimensions

from lib.utils import _get_point
from lib.data_utils import (get_era5_data,
                            get_modis_data,
                            get_elevation_data,
                            get_land_cover_data,
                            get_sen3_data,
                            get_s5p_data,
                            get_cams_data)
from algorithms.ensemble import get_feature_matrix_dict


def add_cams_col(gts, cams_eop):
    COORD_TO_GRID = {}
    dates = np.array(cams_eop.meta_info['day'])

    for i, (date, lat, lon) in enumerate(gts[["Date", "SITE_LATIT", "SITE_LONGI"]].values):
        if (lat, lon) not in COORD_TO_GRID:
            p = _get_point(lat, lon, cams_eop.data["PM2_5"][0], cams_eop.bbox)
            COORD_TO_GRID[(lat, lon)] = p

        p = COORD_TO_GRID[(lat, lon)]
        mask = date == dates
        pm25 = cams_eop.data["PM2_5"][mask][:, p[0], p[1]]
        no2_surface = cams_eop.data["NO2_surface"][mask][:, p[0], p[1]]
        gts.loc[gts.index == i, "PM2_5"] = pm25.mean()
        gts.loc[gts.index == i, "NO2_surface"] = no2_surface.mean()


def filter_gt(pm25_gt, cams_eop):
    add_cams_col(pm25_gt, cams_eop)
    pm25_gt.loc[pm25_gt.PM2_5 > 115, "PM2_5"] = 115

    v1 = pm25_gt.PM2_5
    v2 = pm25_gt.AirQuality
    mask = ((abs(v2 - v1) > v2 * 0.7) & ((v2 > 5) | (v1 > 5)))
    pm25_gt = pm25_gt.loc[~mask]

    return pm25_gt


def get_day_eop(eop, date):
    mask = eop.meta_info["day"] == date
    if not mask.any():
        return None

    out_eop = EOPatch(data={k: v[mask].mean(0, keepdims=True) for k, v in eop.data.items()},
                      meta_info={"date": date},
                      bbox=eop.bbox,
                      )

    return out_eop


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_data(indir, outdir, label, aoi, feature_keys, update=False):
    # Data
    dir = indir/"ground_air_quality" / ("NO2" if label == "NO2" else "PM25")
    gt_path = dir / (os.listdir(dir)[0][:-3] + 'shp')
    gts = gpd.read_file(gt_path)

    path = outdir / ("data_" + aoi + ".pkl")
    if os.path.isfile(path) and not update:
        data = load_pickle(path)
    else:
        data = {"land_cover": get_land_cover_data(indir/("corine" if aoi == "Italy" else ""), outdir),
                "era5": get_era5_data(indir/"era5", outdir),
                "cams": get_cams_data(indir/"CAMS", outdir),
                "modis": get_modis_data(indir/"modis_MCD19A2", outdir),
                "s5p": get_s5p_data(indir/'sentinel5P', outdir),
                "elevation": get_elevation_data(indir, outdir)}
        if aoi != "South_Africa":
            data["s3_eop"] = get_sen3_data(indir/"SEN3", outdir)
        save_pickle(data, path)

    # -----------

    # Day data
    day_data = {"data": [],
                "date": []}
    for date in tqdm.tqdm(gts.Date.unique()):
        _day_data = {
            "land_cover": data["land_cover"],
            "elevation": data["elevation"],
            "era5": get_day_eop(data["era5"], date),
            "modis": get_day_eop(data["modis"], date),
            # "s3": get_day_eop(data["s3"], date),
        }

        if label == "NO2":
            _day_data["s5p"] = get_day_eop(data["s5p"], date)
            if _day_data["s5p"] is None:
                continue
        else:
            _day_data["cams"] = get_day_eop(data["cams"], date)

        if np.any([v is None for v in _day_data.values()]):
            print('Missing datapoint at date:', date, "datasets:", [k for k, v in _day_data.items() if v is None])
            continue

        assert np.all([v is not None for v in _day_data.values()])
        day_data["data"].append(_day_data)
        day_data["date"].append(date)
    # ------------------

    # Target size
    if label == "NO2":
        target_resolution = 1000
        in_eop = day_data["data"][0]["s5p"]
    else:
        target_resolution = 1000 if aoi == "Italy" else 10_000
        in_eop = day_data["data"][0]["cams"]

    target_size = bbox_to_dimensions(in_eop.bbox, target_resolution)[::-1]
    # -----------------

    day_data["feat_dicts"] = []
    day_data["feats"] = []
    for i in range(len(day_data["data"])):
        feat_dict = get_feature_matrix_dict(
            day_data["data"][i], target_size, label, verbose=False)
        day_data["feat_dicts"].append(feat_dict)

        feats = np.stack([feat_dict[k] for k in feature_keys], axis=-1)
        day_data["feats"].append(feats)

    return day_data, gts, target_size
