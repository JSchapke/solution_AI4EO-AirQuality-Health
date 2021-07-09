import numpy as np
from pathlib import Path
from eolearn.core import EOPatch

from lib.data_utils import (get_cams_data,
                            get_era5_data,
                            get_modis_data,
                            get_elevation_data,
                            )


def funnel_day(eop, day):
    eop_day = np.array(eop.meta_info["day"])
    mask = eop_day == day

    return EOPatch(bbox=eop.bbox,
                   data={k: v[mask] for k, v in eop.data.items()},
                   meta_info={"day": eop_day[mask]},
                   # timestamp=eop.timestamp[mask]
                   )


def get_data(indir, outdir):
    indir = Path(indir)
    outdir = Path(outdir)

    cams_eop = get_cams_data(indir/"CAMS", outdir/"CAMS")
    era5_eop = get_era5_data(indir/"era5", outdir/"era5")
    modis_eop = get_modis_data(indir/"modis_MCD19A2", outdir/"modis_MCD19A2")
    elevation_eop = get_elevation_data(indir, outdir)

    data = dict(cams=cams_eop, era5=era5_eop,
                modis=modis_eop, elevation=elevation_eop)
    return data
