import re
import os
import copy
import collections
import numpy as np
from tqdm import tqdm
from eolearn.core import EOPatch, EOTask, LinearWorkflow, FeatureType
from eolearn.io.local_io import ExportToTiff
from sentinelhub import bbox_to_dimensions
from air_quality_and_health_challenge.utils import (load_tiffs,
                                                    reproject_tiff,
                                                    upscale_tiff,
                                                    mask_tiff)


from lib.utils import get_point, _get_point, get_patch, get_coords

NO_DATA_VALUE = -9999.0
OFFSET = 2100


def normalize_datasets(cams_eop, era5_eop, land_cover_eop, modis_eop, elevation_eop):
    state = {
        "PM2_5": dict(mean=cams_eop.data["PM2_5"].mean(), std=cams_eop.data["PM2_5"].std()),
        "NO2_surface": dict(mean=cams_eop.data["NO2_surface"].mean(), std=cams_eop.data["NO2_surface"].std()),
        "wind_v": dict(mean=era5_eop.data["wind_v"].mean(), std=era5_eop.data["wind_v"].std()),
        "wind_u": dict(mean=era5_eop.data["wind_u"].mean(), std=era5_eop.data["wind_u"].std()),
        "specific_rain_water_content": dict(mean=era5_eop.data["specific_rain_water_content"].mean(),
                                            std=era5_eop.data["specific_rain_water_content"].std()),
        "relative_humidity": dict(mean=era5_eop.data["relative_humidity"].mean(),
                                  std=era5_eop.data["relative_humidity"].std()),
        "elevation": dict(mean=elevation_eop.data["elevation"].mean(),
                          std=elevation_eop.data["elevation"].std()),
        "modis_mean": dict(mean=np.nanmean(modis_eop.data["mean"]),
                           std=np.nanmean(modis_eop.data["mean"])),
    }

    def trf(eop, state, key):
        eop = copy.deepcopy(eop)
        eop.data[key] = (eop.data[key] - state["mean"]) / (state["std"] + 1e-8)
        return eop

    cams_eop = trf(cams_eop, state["PM2_5"], "PM2_5")
    cams_eop = trf(cams_eop, state["NO2_surface"], "NO2_surface")
    era5_eop = trf(era5_eop, state["wind_v"], "wind_v")
    era5_eop = trf(era5_eop, state["wind_u"], "wind_u")
    era5_eop = trf(era5_eop, state["specific_rain_water_content"],
                   "specific_rain_water_content")
    era5_eop = trf(era5_eop, state["relative_humidity"], "relative_humidity")
    elevation_eop = trf(elevation_eop, state["elevation"], "elevation")
    modis_eop = trf(modis_eop, state["modis_mean"], "mean")

    datasets = dict(cams_eop=cams_eop,
                    era5_eop=era5_eop,
                    elevation_eop=elevation_eop,
                    modis_eop=modis_eop,
                    land_cover_eop=land_cover_eop)

    return datasets, state


def generate_avg(dir, wdir, data_source, key="PM2_5"):
    avg_dir = wdir/f"AVG_{key}"
    os.makedirs(avg_dir, exist_ok=True)
    exporter = ExportToTiff(key, avg_dir)

    files = os.listdir(dir/key)
    searches = [re.search(r"day\d\d\d\d", f) for f in files]
    days = list(set([s[0] for s in searches if s is not None]))
    for day in days:
        fs = sorted([f for f in files if day in f])
        fname = re.sub(r"_h\d\d", "_h00", fs[0])

        raw = load_tiffs(
            dir/key, (FeatureType.DATA, key), data_source=data_source, tiles=fs, offset=OFFSET)
        raw.data[key] = raw.data[key].mean(0, keepdims=True)
        raw['timestamp'] = raw['timestamp'][:1]

        exporter.execute(raw, filename=fname)


def get_pm25_test_dataset(datasets, dates, state):
    images = []
    for date, time in dates:
        image = upscale(datasets["cams"][date, time])
        images.append(image)

    data = []

    coords_to_grid = {}
    target_size = (a, b)
    for date, time in dates:
        date_data = {"cams": [], "era5": [], "modis": [], "elevation": []}

        day_cams = datasets["cams"][date, time]
        day_era5 = datasets["era5"][date, time]
        day_modis = datasets["modis"][date]

        for a in range(target_size[0]):
            for b in range(target_size[1]):
                if not (a, b) in coords_to_grid:
                    coords_to_grid[(a, b)] = {"cams": (a, b),
                                              "era5": (a, b),
                                              "modis": (a, b),
                                              "elevation": (a, b)}

                a, b = coords_to_grid["cams"]
                date_data["cams"].append(day_cams[a, b])
                a, b = coords_to_grid["era5"]
                date_data["era5"].append(day_cams[a, b])
                a, b = coords_to_grid["cams"]
                date_data["modis"].append(day_cams[a, b])
                a, b = coords_to_grid["cams"]
                date_data["elevation"].append(elevation_eop[a, b])

        pass

    dataset = {
        "X": None,
        "coords": None,
        "images": None,
    }
    return dataset


def get_pm25_train_dataset(pm25_gt,
                           cams_eop,
                           era5_eop,
                           land_cover_eop,
                           modis_eop,
                           elevation_eop,
                           ):

    count = 0
    data = {"gt": [],
            "coords": [],
            "PM2_5": [],
            "NO2_surface": [],
            "wind_v": [],
            "wind_u": [],
            "relative_humidity": [],
            "specific_rain_water_content": [],
            "modis_mean": [],
            "modis_std": [],
            # "land_cover_patch": [],
            "elevation": [],
            }

    COORDS_TO_POINT = collections.defaultdict(dict)

    for date, group in tqdm(pm25_gt.groupby("Date")):
        cams_mask = cams_eop.meta_info["day"] == date
        era5_mask = era5_eop.meta_info["day"] == date
        modis_mask = modis_eop.meta_info["day"] == date

        for i in range(group.shape[0]):
            row = group.iloc[i]
            lat, lon = row["SITE_LATIT"], row["SITE_LONGI"]
            data["gt"].append(row["AirQuality"])
            data["coords"].append((row["SITE_LATIT"], row["SITE_LONGI"]))

            if (lat, lon) not in COORDS_TO_POINT["cams"]:
                y, x = _get_point(
                    lat, lon, cams_eop.data["PM2_5"][0, ..., 0], cams_eop.bbox)
                COORDS_TO_POINT["cams"][(lat, lon)] = (y, x)

                y, x = _get_point(
                    lat, lon, era5_eop.data['wind_v'][0, ..., 0], era5_eop.bbox)
                COORDS_TO_POINT["era5"][(lat, lon)] = (y, x)

                y_ind, x_ind = _get_point(
                    lat, lon, modis_eop.data["mean"][0, ..., modis_mask][..., 0], modis_eop.bbox)
                COORDS_TO_POINT["modis"][(lat, lon)] = (y_ind, x_ind)

                y, x = _get_point(
                    lat, lon, elevation_eop.data["elevation"][0, ..., 0], elevation_eop.bbox)
                COORDS_TO_POINT["elevation"][(lat, lon)] = (y, x)

            y, x = COORDS_TO_POINT["cams"][(lat, lon)]
            data["PM2_5"].append(cams_eop.data["PM2_5"]
                                 [cams_mask, y, x].mean())
            data["NO2_surface"].append(
                cams_eop.data["NO2_surface"][cams_mask, y, x].mean())

            keys = list(era5_eop.data.keys())
            y, x = COORDS_TO_POINT["era5"][(lat, lon)]
            [data[k].append(era5_eop.data[k][era5_mask, y, x].mean())
             for k in keys]

            y_ind, x_ind = COORDS_TO_POINT["modis"][(lat, lon)]
            data["modis_mean"].append(
                modis_eop.data["mean"][0, y_ind, x_ind, modis_mask][..., 0])
            data["modis_std"].append(
                modis_eop.data["std"][0, y_ind, x_ind, modis_mask][..., 0])

            y, x = COORDS_TO_POINT["elevation"][(lat, lon)]
            data["elevation"].append(
                elevation_eop.data["elevation"][0, y, x, 0])

            # land_cover_patch = get_patch(row, land_cover_eop, 5)
            # data["land_cover_patch"].append(land_cover_patch)

    data = {k: np.array(v) for k, v in data.items()}
    return data


def upscale_image(dir, wdir, date, time, bbox):

    # Base file
    cams_im = load_tiffs(dir/"CAMS/PM2_5",  (FeatureType.DATA, 'PM2_5'),
                         filename=f"CAMS_PM2_5_day{date}_{time}.tif", data_source="cams")
    target_size = bbox_to_dimensions(cams_im.bbox, 1000)

    # this function applies upscaling to first channel only, change if needed
    upscale_tiff(dir/"CAMS/PM2_5"/f"CAMS_PM2_5_day{date}_{time}.tif",
                 wdir/f'TEST_CAMS_PM25_day{date}_{time}.tif',
                 target_size)

    bbox.to_crs(epsg=4326, inplace=True)
    bbox.to_file(wdir/'bbox-wgs84.shp', driver='ESRI Shapefile')

    mask_tiff(wdir/'bbox-wgs84.shp',
              wdir/f'TEST_CAMS_PM25_day{date}_{time}.tif',
              wdir/f'TEST_CROPPED_CAMS_PM25_day{date}_{time}.tif')

    output_img = load_tiffs(wdir, (FeatureType.DATA, 'PM2_5'), data_source="cams",
                            filename=f'TEST_CROPPED_CAMS_PM25_day{date}_{time}.tif')
    return output_img


def get_pm25_test_dataset_for_date(dir,
                                   wdir,
                                   date,
                                   time,
                                   bbox,
                                   cams_eop,
                                   era5_eop,
                                   land_cover_eop,
                                   modis_eop,
                                   elevation_eop):

    test_img = upscale_image(dir, wdir, date, time, bbox)
    coords, positions = get_coords(test_img)

    # Other data
    dates = np.array(cams_eop.meta_info['day'])
    day_mask = dates == date
    timestamp = np.array(cams_eop.timestamp)[day_mask]
    time_mask = [(e.hour == int(time[1:])) for e in timestamp]

    cams = {"PM2_5": cams_eop.data["PM2_5"][day_mask].squeeze(),
            "NO2_surface": cams_eop.data["NO2_surface"][day_mask].squeeze(),
            "bbox": cams_eop.bbox}

    dates = np.array(era5_eop.meta_info['day'])
    times = np.array([t.hour for t in era5_eop.timestamp])
    mask = (dates == date) & (times == int(time[1:]))
    era5 = {}
    for key, value in era5_eop.data.items():
        era5[key] = value[mask].squeeze()

    dataset = {"coord": [],
               "position": [],
               "pm25": [],
               "no2_surface": [],
               "wind_v": [],
               "wind_u": [],
               "relative_humidity": [],
               "specific_rain_water_content": [],
               # "modis_mean": [],
               # "modis_std": [],
               # "land_cover_patch": [],
               }

    for i, (lon, lat) in enumerate(tqdm(coords)):
        dataset["coord"].append((lat, lon))
        dataset["position"].append(positions[i])

        p = _get_point(lat, lon, cams["PM2_5"][0], cams["bbox"])
        dataset["pm25"].append(cams["PM2_5"][:, p[0], p[1]])

        p = _get_point(lat, lon, era5["wind_v"], era5_eop.bbox)
        for key in era5.keys():
            dataset[key].append(era5[key][p[0], p[1]])

    return dataset, test_img


#############################################################################################

def process_pm25_datasets(pm25_gt,
                          cams_eop,
                          era5_eop,
                          land_cover_eop,
                          modis_eop,
                          dem_eop):
    # Filter GT data
    add_cams_col(pm25_gt, cams_eop)
    pm25_gt.loc[pm25_gt.PM2_5 > 115, "PM2_5"] = 115

    v1 = pm25_gt.PM2_5
    v2 = pm25_gt.AirQuality
    mask = ((abs(v2 - v1) > v2 * 0.7) & ((v2 > 5) | (v1 > 5)))
    pm25_gt = pm25_gt.loc[~mask]


#############################################################################################
# CAMS
def get_cams_data(dir, wdir, datetime=None, bbox_path=None, average=True):
    filenames = dict(NO2_surface=None, PM2_5=None)
    if datetime is not None:
        filenames["PM2_5"] = f"CAMS_PM2_5_day{datetime[0]}_{datetime[1]}.tif"
        filenames["NO2_surface"] = f"CAMS_NO2_day{datetime[0]}_{datetime[1]}.tif"

    cams_eop = None
    for key in ['PM2_5', 'NO2_surface']:
        eop_file = load_tiffs(
            datapath=dir/key,
            feature=(FeatureType.DATA, key),
            data_source="cams",
            filename=filenames[key])

        if cams_eop is None:
            cams_eop = eop_file
        else:
            cams_eop.data[key] = eop_file.data[key]
    return cams_eop


#############################################################################################
# ERA 5
def get_era5_data(dir, wdir, datetime=None):
    keys = "specific_rain_water_content", "wind_v", "wind_u", "relative_humidity"
    filenames = {k: None for k in keys}
    if datetime:
        filenames["specific_rain_water_content"] = f"ERA5_srwc_day{datetime[0]}_{datetime[1]}.tif"
        filenames["relative_humidity"] = f"ERA5_rh_day{datetime[0]}_{datetime[1]}.tif"
        filenames["wind_u"] = f"ERA5_u_day{datetime[0]}_{datetime[1]}.tif"
        filenames["wind_v"] = f"ERA5_v_day{datetime[0]}_{datetime[1]}.tif"

    era5_eop = None
    for key in keys:
        eop = load_tiffs(dir/key, (FeatureType.DATA, key), offset=OFFSET,
                         image_dtype=np.float32, data_source='era5', filename=filenames[key])

        if era5_eop is None:
            era5_eop = eop
        else:
            era5_eop.data[key] = eop.data[key]
    return era5_eop


#############################################################################################
# SENTINEL-5P
def get_s5p_data(dir, wdir, datetime=None):
    keys = ["NO2", "UV_AEROSOL_INDEX"]
    filenames = {k: None for k in keys}
    if datetime:
        filenames["NO2"] = [f for f in os.listdir(dir/"NO2")
                            if f"day{datetime[0]}_T{datetime[1][1:]}.tif" in f][0]
        filenames["UV_AEROSOL_INDEX"] = [f for f in os.listdir(dir/"UV_AEROSOL_INDEX")
                                         if f"day{datetime[0]}_T{datetime[1][1:]}.tif" in f][0]
    s5p_eop = None
    for key in keys:
        eop = load_tiffs(dir/key, (FeatureType.DATA, key), data_source="s5p",
                         filename=filenames[key])
        eop.data[key][eop.data[key] == NO_DATA_VALUE] = np.nan
        min_q_a = 50
        eop.data[key][..., 0][eop.data[key][..., 1] <= min_q_a] = np.nan

        if s5p_eop is None:
            s5p_eop = eop
        else:
            s5p_eop.data[key] = eop.data[key]

    no2_grid = s5p_eop.data["NO2"]
    uv_grid = s5p_eop.data["UV_AEROSOL_INDEX"]
    if "South_Africa" in str(dir) and no2_grid.shape[0] != uv_grid.shape[0]:
        n = no2_grid.shape[0] - uv_grid.shape[0]
        uv_grid = np.concatenate(
            [np.full((n, *uv_grid.shape[1:]), np.nan), uv_grid], axis=0)
        s5p_eop.data["UV_AEROSOL_INDEX"] = uv_grid

    return s5p_eop


#############################################################################################
# SENTINEL-3
def get_sen3_data(dir, wdir, datetime=None):
    if datetime:
        filename = [f for f in os.listdir(dir) if str(datetime[0]) in f][0]
        return load_tiffs(datapath=dir, feature=(FeatureType.DATA, 'S3'),
                          filename=filename, image_dtype=np.float32, data_source='s3')

    s3_eops = [load_tiffs(datapath=dir,
                          feature=(FeatureType.DATA, 'S3'),
                          filename=filename,
                          image_dtype=np.float32,
                          data_source='s3')
               for filename in sorted(os.listdir(dir))]

    days = np.stack([e.meta_info["day"] for e in s3_eops], 0).reshape(-1)
    data = np.concatenate([e.data["S3"] for e in s3_eops], 0)
    s3_eop = s3_eops[0]
    s3_eop.data['S3'] = data
    s3_eop.meta_info['day'] = days

    return s3_eops


#############################################################################################
# LAND COVER
def get_land_cover_data(dir, wdir):
    fname = 'land_cover.wgs84.tif'

    if "Italy" in str(dir):
        if not os.path.isfile(wdir/fname):
            reproject_tiff(dir/'U2018_CLC2018_V2020_20u1_North_Italy.tif',
                           wdir/fname, dst_crs='EPSG:4326')

        land_cover_eop_wgs84 = load_tiffs(datapath=wdir,
                                          feature=(
                                              FeatureType.MASK_TIMELESS, 'CORINE'),
                                          filename=fname,
                                          image_dtype=np.uint8,
                                          no_data_value=128)

    elif "South_Africa" in str(dir):
        land_cover_eop_wgs84 = load_tiffs(datapath=dir/"Land_cover",
                                          feature=(
                                              FeatureType.MASK_TIMELESS, 'CORINE'),
                                          filename="SANLC_2018_africa.tif",
                                          image_dtype=np.uint8,
                                          no_data_value=128)
    else:
        assert "California" in str(dir)
        land_cover_eop_wgs84 = load_tiffs(datapath=dir/"Land_cover",
                                          feature=(
                                              FeatureType.MASK_TIMELESS, 'CORINE'),
                                          filename="Copernicus_Global_Land_Cover_W140N40_california.tif",
                                          image_dtype=np.uint8,
                                          no_data_value=128)

    return land_cover_eop_wgs84


#############################################################################################
# MODIS
def get_modis_data(dir, wdir, datetime=None):
    if datetime:
        modis_eop = load_tiffs(datapath=dir/'MCD19A2_AOD',
                               feature=(FeatureType.DATA, 'AOD'),
                               filename=f"MCD19A2_day{datetime[0]}.tif",
                               image_dtype=np.float32, data_source='modis')
        modis_eop.data["mean"] = np.nanmean(
            modis_eop.data["AOD"], keepdims=True, axis=-1)

        return modis_eop

    else:
        modis_eops = [load_tiffs(datapath=dir/'MCD19A2_AOD',
                                 feature=(FeatureType.DATA, 'AOD'),
                                 filename=filename,
                                 image_dtype=np.float32,
                                 data_source='modis')
                      for filename in sorted(os.listdir(dir/'MCD19A2_AOD'))]

        aod_mean = np.stack([np.nanmean(eo.data['AOD'], -1)
                             for eo in modis_eops], -1)
        aod_std = np.stack([np.nanstd(eo.data['AOD'], -1)
                            for eo in modis_eops], -1)
    day = np.concatenate([eo.meta_info['day'] for eo in modis_eops])
    modis_eop = EOPatch(data={"mean": aod_mean, "std": aod_std},
                        bbox=modis_eops[0].bbox, meta_info={"day": day})
    modis_eop.data = {k: np.swapaxes(v, 0, 3)
                      for k, v in modis_eop.data.items()}

    return modis_eop


#############################################################################################
# DEM/DSM
def get_elevation_data(dir, wdir):
    if "Italy" in str(dir):
        filename = "eu_dem_v11_North_Italy.tif"
    elif "California" in str(dir):
        filename = "COP_DSM_california_GLO-30.tif"
    elif "South_Africa" in str(dir) or "SouthAfrica" in str(dir):
        filename = 'COP_DSM_south_africa_GLO-30.tif'
    else:
        raise Exception(str(dir))

    elevation_eop = load_tiffs(datapath=dir,
                               feature=(FeatureType.DATA, 'elevation'),
                               filename=filename,
                               image_dtype=np.float32)
    return elevation_eop


#############################################################################################
