import os
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
# Use time only between 9am to 5pm
times = [t for t in times if t.time() >= time(9, 0) and t.time() <= time(17, 0)]
chunk_num = 50
chunked_times = np.array_split(times, chunk_num)
def save_single_chunk(time_chunk, index):
    if os.path.exists('f/data/climate_hack/chunk_{index}'):
        return
    data_slice = dataset.loc[{'time': time_chunk}]["data"].values
    data_dict = {}
    for i, time in enumerate(time_chunk):
        data_dict[str(time)] = data_slice[i]
    np.savez_compressed(f'/data/climate_hack/chunk_{index}', **data_dict)

pool = Pool(8)
pool.starmap(save_single_chunk, zip(chunked_times, range(len(chunked_times))))
