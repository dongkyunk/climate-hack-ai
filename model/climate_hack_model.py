import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from model.transformer import ImageGPT
from pytorch_msssim import MS_SSIM


class ClimateHackModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackModel, self).__init__()
        self.cfg = cfg
        self.model = ImageGPT()
        self.criterion = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=24)
        self.lr = self.cfg.trainer_cfg.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        osgb_data, input_image, target_image = batch
        output_image = self(input_image)
        ssim_loss = 1 - self.criterion(output_image, target_image)
        return dict(
            loss=ssim_loss,
            output_image=output_image.detach().view(-1, 1, 64, 64)/1024,
            target_image=target_image.detach().view(-1, 1, 64, 64)/1024,
        )

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.log_dict({
            "val_loss": outputs["loss"], 
        }, prog_bar=True, sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_image(outputs["output_image"], "output_image.png")
            save_image(outputs["target_image"], "target_image.png")


    def configure_optimizers(self):
        optimizer = MADGRAD(self.model.parameters(), lr=self.lr)  
        return optimizer