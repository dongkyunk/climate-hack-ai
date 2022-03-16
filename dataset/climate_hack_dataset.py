from calendar import c
import os
from re import S
from tkinter import E
import torch
import numpy as np
from datetime import timedelta
from random import randrange
from torch.utils.data import IterableDataset


class ClimateHackDataset(IterableDataset):
    def __init__(self, cfg, times, start_times) -> None:
        super().__init__()
        self.cfg = cfg
        self.times = times
        self.start_times = start_times
        self.sorted_times = self.times.sort_index()
        self.cached_data = []
        self.chunk_num = 0

    def _load_image(self, x_0, x_1, y_0, y_1, time):
        if str(time) in self.cached_data[0].files:
            return self.cached_data[0][str(time)][x_0:x_1, y_0:y_1]
        elif str(time) in self.cached_data[1].files:
            return self.cached_data[1][str(time)][x_0:x_1, y_0:y_1]
        else:
            self.cached_data.pop(0)
            self.cached_data.append(np.load(os.path.join(self.cfg.path_cfg.data_dir, f'chunk_{self.chunk_num}.npz')))
            self.chunk_num += 1

    def __iter__(self):
        self.chunk_num = 0
        self.cached_data.append(np.load(os.path.join(self.cfg.path_cfg.data_dir, f'chunk_{self.chunk_num}.npz')))
        self.chunk_num += 1
        self.cached_data.append(np.load(os.path.join(self.cfg.path_cfg.data_dir, f'chunk_{self.chunk_num}.npz')))
        self.chunk_num += 1

        for i in range(len(self.start_times)):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:  
                rank = worker_info.id
                if i % worker_info.num_workers != rank:
                    continue            

            start_time = self.start_times.index[i]

            # Pick location
            # rand_x = randrange(0, 891 - 128)
            # rand_y = randrange(0, 1843 - 128)

            # # Load input
            # try:
            #     image = self._load_image(rand_x, rand_x + 128, rand_y, rand_y + 128, start_time)
            #     if image is None:
            #         continue
            #     yield image
            # except:
            #     continue

            # Pick time slice
            time_slice = self.sorted_times[start_time:start_time+timedelta(hours=2, minutes=55)]
            if len(time_slice) != 36:
                continue

            # Slice input and target
            input_slice = time_slice[:12]
            target_slice = time_slice[12:]
            # target_slice = time_slice[12:24]

            # Pick location
            rand_x = randrange(0, 891 - 128)
            rand_y = randrange(0, 1843 - 128)

            # Load input
            input_image = []
            try:
                for time in input_slice.index:
                    image = self._load_image(rand_x, rand_x + 128, rand_y, rand_y + 128, time)
                    # image = self._load_image(rand_x+32, rand_x+96, rand_y+32, rand_y+96, time)
                    input_image.append(image)
                input_image = np.stack(input_image, axis=0)
            except:
                continue

            # yield input_image
                      
            # Load target
            target_image = []
            try:
                for time in target_slice.index:
                    # image = self._load_image(rand_x+32, rand_x+96, rand_y+32, rand_y+96, time)
                    image = self._load_image(rand_x, rand_x + 128, rand_y, rand_y + 128, time)
                    target_image.append(image)
                target_image = np.stack(target_image, axis=0)
            except:
                continue

            yield input_image, target_image


