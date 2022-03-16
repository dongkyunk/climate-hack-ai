import pandas as pd
import xarray as xr
import pytorch_lightning as pl

from datetime import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import Config
from dataset.climate_hack_dataset import ClimateHackDataset


class ClimateHackDatamodule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super(ClimateHackDatamodule, self).__init__()
        self.cfg = cfg
        dataset = xr.open_dataset(
            self.cfg.path_cfg.xarr_dataset_path,
            engine="zarr",
            chunks="auto",  # Load the data as a Dask array
        )
        self.times = dataset.get_index("time")

        # Use time only between 9am to 5pm
        self.times = [t for t in self.times if t.time() >= time(
            9, 0) and t.time() <= time(17, 0)]
        self.start_times = [t for t in self.times if t.time() >= time(
            9, 0) and t.time() <= time(14, 0)]
        self.times = pd.Series(range(len(self.times)), index=self.times)
        self.start_times = pd.Series(
            range(len(self.start_times)), index=self.start_times)

    def setup(self, stage):
        val_times = self.start_times[:1000]
        self.train_dataset = ClimateHackDataset(self.cfg, self.times, self.start_times)
        self.val_dataset = ClimateHackDataset(self.cfg, self.times, val_times)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.trainer_cfg.train_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.trainer_cfg.val_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)