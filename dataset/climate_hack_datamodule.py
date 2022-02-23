import pandas as pd
import pytorch_lightning as pl
import xarray as xr
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import time

from config import Config
from dataset.climate_hack_dataset import ClimateHackDataset


def none_chuck_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ClimateHackDatamodule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super(ClimateHackDatamodule, self).__init__()
        self.cfg = cfg
        self.dataset = xr.open_dataset(
            self.cfg.path_cfg.data_path, 
            engine="zarr",
            chunks="auto",  # Load the data as a Dask array
        )
        self.times = self.dataset.get_index("time")
        # Use time only between 9am to 2pm
        self.times = [t for t in self.times if t.time() >= time(9, 0) and t.time() <= time(14, 0)]
        self.times = pd.Series(range(len(self.times)) , index=self.times)

    def setup(self, stage):
        train_times, val_times = train_test_split(self.times, test_size=0.2, random_state=42)
        self.train_dataset = ClimateHackDataset(self.dataset, train_times)
        self.val_dataset = ClimateHackDataset(self.dataset, val_times)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.trainer_cfg.train_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=True, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers,
                          collate_fn=none_chuck_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.trainer_cfg.val_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers,
                          collate_fn=none_chuck_collate)
