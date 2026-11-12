"""Example 3: Geospatial & Environmental Analysis with xarray DataArrays and Datasets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr
from pymu_eval import Parser, eval_expr


def create_sample_earth_observation_dataset() -> xr.Dataset:
    """Create a synthetic multi-spectral & meteorological xarray.Dataset."""
    rng = np.random.default_rng(42)
    y = np.linspace(52.0, 52.3, 4)
    x = np.linspace(-1.5, -1.1, 5)
    time = ["2026-06-01", "2026-07-01"]

    shape = (len(time), len(y), len(x))
    blue = rng.uniform(0.02, 0.10, size=shape)
    green = rng.uniform(0.04, 0.15, size=shape)
    red = rng.uniform(0.03, 0.25, size=shape)
    nir = rng.uniform(0.20, 0.65, size=shape)
    swir = rng.uniform(0.10, 0.40, size=shape)

    # Inject a simulated cloud pixel at (time=0, y=1, x=2)
    cloud_mask = np.zeros(shape, dtype=int)
    cloud_mask[0, 1, 2] = 1

    # Surface temperature (Celsius) and relative humidity (%)
    temp_c = rng.uniform(15.0, 32.0, size=shape)
    rel_humidity = rng.uniform(40.0, 85.0, size=shape)

    return xr.Dataset(
        data_vars={
            "blue": (("time", "lat", "lon"), blue),
            "green": (("time", "lat", "lon"), green),
            "red": (("time", "lat", "lon"), red),
            "nir": (("time", "lat", "lon"), nir),
            "swir": (("time", "lat", "lon"), swir),
            "cloud": (("time", "lat", "lon"), cloud_mask),
            "temp_c": (("time", "lat", "lon"), temp_c),
            "rh": (("time", "lat", "lon"), rel_humidity),
        },
        coords={
            "time": time,
            "lat": y,
            "lon": x,
        },
    )


def main() -> None:
    ds = create_sample_earth_observation_dataset()
    print("=== Input xarray.Dataset ===")
    print(ds, "\n")

    print("=== 1. Cloud-Masked NDVI & EVI directly from xarray.Dataset ===")
    # Evaluate cloud-masked NDVI using muParser ternary operator (? :)
    ds["ndvi"] = eval_expr(
        "cloud == 0 ? (nir - red) / (nir + red) : nan",
        ds,
        nan=np.nan,
    )
    # Enhanced Vegetation Index (EVI)
    ds["evi"] = eval_expr(
        "cloud == 0 ? 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0) : nan",
        ds,
        nan=np.nan,
    )
    print("Cloud-masked NDVI (time=2026-06-01):")
    print(ds["ndvi"].sel(time="2026-06-01").to_numpy().round(3), "\n")

    print("=== 2. Multi-Threshold Classification with Nested Ternary (? :) ===")
    # 0 = Cloud/Invalid, 1 = Bare/Sparse (NDVI < 0.2), 2 = Moderate Veg (0.2..0.5), 3 = Dense Veg (>= 0.5)
    ds["veg_class"] = eval_expr(
        "cloud != 0 ? 0 : ndvi < 0.2 ? 1 : ndvi < 0.5 ? 2 : 3",
        ds,
    )
    print("Vegetation Class (time=2026-06-01):")
    print(ds["veg_class"].sel(time="2026-06-01").to_numpy(), "\n")

    print("=== 3. Meteorological Vapor Pressure Deficit (VPD in kPa) ===")
    p = Parser()
    p.define_dataset(ds)
    # Saturation vapor pressure (es) & VPD = es * (1 - rh / 100)
    p.set_expr("0.61078 * exp((17.27 * temp_c) / (temp_c + 237.3)) * (1.0 - rh / 100.0)")
    ds["vpd_kpa"] = p.eval()
    print("Vapor Pressure Deficit (kPa) mean over space:", ds["vpd_kpa"].mean(dim=("lat", "lon")).to_numpy().round(3))


if __name__ == "__main__":
    main()
