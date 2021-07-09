import pickle
import os
import random
import cv2

from algorithms.mlp import MLP
from lib.data_utils import *


def get_feature_matrix_dict(data, target_size, label, verbose=True):
    target_size = target_size[::-1]
    if verbose:
        print("Data sizes:",
              "\nmodis =", data['modis'].data['mean'][0, ..., 0].shape,
              "\nelevation =", data['elevation'].data['elevation'][0, ..., 0].shape,
              "\nera5 =", data['era5'].data['wind_v'][0, ..., 0].shape)
        if "s3" in data:
            print("s3 =", data['s3'].data['S3'][0].shape)
        if label == "NO2":
            print("s5p_no2 =", data['s5p'].data['NO2'][0, ..., 0].shape,
                  "\ns5p_uv =", data['s5p'].data['UV_AEROSOL_INDEX'][0, ..., 0].shape,)
        else:
            print("cams_pm25 =", data['cams'].data['PM2_5'][0, ..., 0].shape,
                  "\ncams_no2s =", data['cams'].data['NO2_surface'][0, ..., 0].shape,)

    feature_matrix_dict = {}
    if label == "NO2":
        s5p_no2 = data['s5p'].data['NO2'][0, ..., 0]
        s5p_uv = data['s5p'].data['UV_AEROSOL_INDEX'][0, ..., 0]
        feats = {"no2_native":  cv2.resize(s5p_no2, target_size, interpolation=cv2.INTER_NEAREST),
                 "no2_target":  cv2.resize(s5p_no2, target_size, interpolation=cv2.INTER_LINEAR),
                 "uv_native":  cv2.resize(s5p_uv, target_size, interpolation=cv2.INTER_NEAREST),
                 "uv_target":  cv2.resize(s5p_uv, target_size, interpolation=cv2.INTER_LINEAR), }
        feats["no2_diff"] = feats["no2_target"] - feats["no2_native"]
        feats["uv_diff"] = feats["uv_target"] - feats["uv_native"]
        feature_matrix_dict.update(feats)
        native_size = s5p_no2.shape
    else:
        cams_pm25 = data['cams'].data['PM2_5'][0, ..., 0]
        cams_no2s = data['cams'].data['NO2_surface'][0, ..., 0]
        feats = {"pm25_native":  cv2.resize(cams_pm25, target_size, interpolation=cv2.INTER_NEAREST),
                 "pm25_target":  cv2.resize(cams_pm25, target_size, interpolation=cv2.INTER_LINEAR),
                 "no2_surface_native":  cv2.resize(cams_no2s, target_size, interpolation=cv2.INTER_NEAREST),
                 "no2_surface_target":  cv2.resize(cams_no2s, target_size, interpolation=cv2.INTER_LINEAR), }
        feats["pm25_diff"] = feats["pm25_target"] - feats["pm25_native"]
        feats["no2_surface_diff"] = feats["no2_surface_target"] - \
            feats["no2_surface_native"]
        feature_matrix_dict.update(feats)
        native_size = cams_pm25.shape

    modis = data['modis'].data['mean'][0, ..., 0]
    modis_native = cv2.resize(
        modis, native_size, interpolation=cv2.INTER_LINEAR)
    elevation = data['elevation'].data['elevation'][0, ..., 0]
    elevation_native = cv2.resize(
        elevation, native_size, interpolation=cv2.INTER_LINEAR)

    era5 = {k: v[0, ..., 0] for k, v in data['era5'].data.items()}
    era5_target = {k: cv2.resize(
        v, target_size, interpolation=cv2.INTER_LINEAR) for k, v in era5.items()}
    era5_native_native = {k: cv2.resize(
        v, native_size, interpolation=cv2.INTER_LINEAR) for k, v in era5.items()}
    era5_native_target = {k: cv2.resize(v, target_size, interpolation=cv2.INTER_NEAREST)
                          for k, v in era5_native_native.items()}

    feats = {"modis_target": cv2.resize(modis, target_size, interpolation=cv2.INTER_LINEAR),
             "modis_native": cv2.resize(modis_native, target_size, interpolation=cv2.INTER_NEAREST),
             "elevation_target": cv2.resize(elevation, target_size, interpolation=cv2.INTER_LINEAR),
             "elevation_native": cv2.resize(elevation_native, target_size, interpolation=cv2.INTER_NEAREST),
             **{f"{k}_native": v for k, v in era5_native_target.items()},
             **{f"{k}_target": v for k, v in era5_target.items()},
             **{f"{k}_diff": v - era5_native_target[k] for k, v in era5_target.items()},
             }
    feats["modis_diff"] = feats["modis_target"] - feats["modis_native"]
    feats["elevation_diff"] = feats["elevation_target"] - \
        feats["elevation_native"]
    feature_matrix_dict.update(feats)
    return feature_matrix_dict


class Ensemble:
    def __init__(self, config, data_dir, write_dir):
        self.config = config
        self.data_dir = data_dir
        self.write_dir = write_dir
        self.label = config["label"]
        assert self.label in ["PM2.5", "NO2"]
        self._data = None
        os.makedirs(write_dir, exist_ok=True)

        indim = self.config["indim"]
        outdim = self.config["outdim"]
        self.models = {k: MLP(params, indim, outdim)
                       for k, params in self.config["models"].items()}

    def reload(self):
        with open(self.config["weights_path"], "rb") as f:
            weights = pickle.load(f)

        for key in weights.keys():
            self.models[key].set_weights(weights[key])

    def save(self):
        state = {k: model.get_weights()
                 for k, model in self.models.items()}
        os.makedirs(os.path.dirname(self.config["weights_path"]), exist_ok=True)
        with open(self.config["weights_path"], "wb") as f:
            pickle.dump(state, f)

    def get_data(self, datetime=None):
        dir = self.data_dir
        wdir = self.write_dir
        if self._data is None:
            self._data = {  # "land_cover": get_land_cover_data(dir, wdir),
                "elevation": get_elevation_data(dir, wdir)}

        data = self._data.copy()
        if os.path.isdir(dir/"SEN3"):
            data["s3"] = get_sen3_data(dir/"SEN3", wdir, datetime=datetime)

        data["era5"] = get_era5_data(dir/"era5", wdir, datetime=datetime)
        data["modis"] = get_modis_data(
            dir/"modis_MCD19A2", wdir, datetime=datetime)

        if self.label == "NO2":
            data["s5p"] = get_s5p_data(
                dir/"sentinel5P", wdir, datetime=datetime)
        else:
            data["cams"] = get_cams_data(dir/"CAMS", wdir, datetime=datetime)

        return data

    def predict(self, datetime, target_size, data=None, verbose=True):
        if data is None:
            data = self.get_data(datetime)

        feature_matrix_dict = get_feature_matrix_dict(
            data, target_size, self.label, verbose=verbose)

        feats = np.stack([feature_matrix_dict[k]
                          for k in self.config['feature_keys']], axis=-1)
        X = feats.reshape((-1, feats.shape[-1]))

        prediction = np.mean([model.predict(X, preprocess=True)
                              for model in self.models.values()], axis=0)
        prediction = prediction.reshape(feats.shape[:-1])

        return prediction

    def train(self, X, Y, coords=None, mu=None, sigma=None):
        for model in self.models.values():
            test_coords = random.sample(np.unique(coords, axis=0).tolist(), 10)
            test_mask = []
            for coord in coords:
                test_mask.append(coord.tolist() in test_coords)
            test_mask = np.array(test_mask)

            train_X, test_X = X[~test_mask], X[test_mask]
            train_Y, test_Y = Y[~test_mask], Y[test_mask]
            print("train_X.shape:", train_X.shape)
            print("test_X.shape:", test_X.shape)

            model.train(train_X, train_Y, X_eval=test_X,
                        y_eval=test_Y, mu=mu, sigma=sigma)
