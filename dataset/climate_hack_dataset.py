import torch
import numpy as np
from numpy import float32
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from random import randrange


class ClimateHackDataset(Dataset):
    def __init__(self, xr_dataset, times) -> None:
        super().__init__()
        self.dataset = xr_dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index):
        # Pick time slice
        curr_time = self.times.index[index]
        print(curr_time)
        data_slice = self.dataset.loc[{"time": slice(curr_time, curr_time + timedelta(hours=3, minutes=0))}]
        input_slice = data_slice.isel(time=slice(0, 12))
        target_slice = data_slice.isel(time=slice(12, 36))

        # Pick location roughly over the mainland UK
        rand_x = randrange(0, 1843 - 128)
        rand_y = randrange(0, 891 - 128)
        input_slice = input_slice.isel(x=slice(rand_x, rand_x + 128), y=slice(rand_y, rand_y + 128))

        # get the OSGB coordinate data
        osgb_data = np.stack([
                input_slice["x_osgb"].values.astype(float32),
                input_slice["y_osgb"].values.astype(float32),
        ])

        input_image = input_slice["data"].values.astype(float32)

        # get the target output
        target_slice = target_slice.isel(x=slice(rand_x + 32, rand_x + 96), y=slice(rand_y + 32, rand_y + 96))
        target_image = target_slice["data"].values.astype(float32)

        is_corrupt = False
        is_corrupt = is_corrupt or osgb_data.shape != (2, 128, 128)
        is_corrupt = is_corrupt or input_image.shape != (12, 128, 128)
        is_corrupt = is_corrupt or target_image.shape != (24, 64, 64)
        if is_corrupt:
            return None

        return osgb_data, input_image, target_image


