import xarray as xr
import numpy as np
from ray.util.multiprocessing import Pool
from datetime import time

dataset = xr.open_dataset(
    "/home/dongkyun/Desktop/Other/climate-hack-ai/data/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr", 
    engine="zarr",
    chunks="auto",  # Load the data as a Dask array
)

times = dataset.get_index("time")
# Use time only between 9am to 2pm
times = [t for t in times if t.time() >= time(9, 0) and t.time() <= time(14, 0)]

def save_single_row(time):
    np.save(f'/data/climate_hack/{str(time)}.npy', dataset.sel(time=time)['data'])

pool = Pool(14)
pool.map(save_single_row, times)


