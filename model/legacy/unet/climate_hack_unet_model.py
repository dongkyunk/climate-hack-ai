import re
import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.unet.block import Unet
from model.unet.discriminator import NLayerDiscriminator, weights_init, hinge_d_loss
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from transformers.optimization import get_cosine_schedule_with_warmup
from model.loss import MS_SSIM, VGGPerceptualLoss


class ClimateHackUnetModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackUnetModel, self).__init__()
        self.cfg = cfg
        self.model = Unet()
        self.discriminator = NLayerDiscriminator(36).apply(weights_init)
        self.criterion_disc = hinge_d_loss
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ssim = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=24)
        self.criterion_perceptual = VGGPerceptualLoss()
        self.lr = self.cfg.trainer_cfg.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, optimizer_idx):
        input_image, target_image = batch
        input_image, target_image = input_image.float(), target_image.float()

        output_image = self.forward(input_image/1024)

        if optimizer_idx == 0:
            logits_fake = self.discriminator(torch.cat([input_image, output_image], dim=1))
            loss_g = -torch.mean(logits_fake)
            loss_l1 = self.criterion_l1(output_image/1024, target_image/1024)*1e2
            loss_ssim = 1 - self.criterion_ssim(output_image.squeeze(), target_image.squeeze())*1e-1
            loss_perceptual = self.criterion_perceptual(output_image.view(-1, 1, 128, 128)/1024, target_image.view(-1, 1, 128, 128)/1024)
            loss = loss_l1 + loss_ssim + loss_perceptual + loss_g
            return dict(
                loss=loss,
                logs=dict(
                    loss=loss.detach(),
                    loss_l1=loss_l1.detach(),
                    loss_ssim=loss_ssim.detach(),
                    loss_perceptual=loss_perceptual.detach(),
                    loss_g=loss_g.detach(),
                ),
                images=dict(
                    output_image=(output_image.view(-1, 1, 128, 128)/1024).detach(),
                    target_image=(target_image.view(-1, 1, 128, 128)/1024).detach(),
                )
            )
        elif optimizer_idx == 1:
            logits_real = self.discriminator(torch.cat([input_image, target_image], dim=1).detach())
            logits_fake = self.discriminator(torch.cat([input_image, output_image], dim=1).detach())
            loss_d = self.criterion_disc(logits_real, logits_fake)
            return dict(
                loss=loss_d,
                logs=dict(
                    loss_d=loss_d.detach(),
                ),
                images=dict(
                    output_image=(output_image.view(-1, 1, 128, 128)/1024).detach(),
                    target_image=(target_image.view(-1, 1, 128, 128)/1024).detach(),
                )
            )

    def training_step(self, batch, batch_idx, optimizer_idx):
        outputs = self.shared_step(batch, batch_idx, optimizer_idx)
        self.log_dict(outputs["logs"], prog_bar=True, sync_dist=True)
        if batch_idx % 50 == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], "train_output_image.png")
            save_image(outputs["images"]["target_image"], "train_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["train/output_image"].upload("train_output_image.png")
                self.logger.experiment["train/target_image"].upload("train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx, 0)
        outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        self.log_dict(outputs["logs"], prog_bar=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], "val_output_image.png")
            save_image(outputs["images"]["target_image"], "val_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["val/output_image"].upload("val_output_image.png")
                self.logger.experiment["val/target_image"].upload("val_target_image.png")

    def configure_optimizers(self):
        # optimizer = MADGRAD(self.parameters(), lr=self.lr)  
        # return optimizer
        g_optimizer = MADGRAD(self.model.parameters(), lr=self.lr)
        d_optimizer = MADGRAD(self.discriminator.parameters(), lr=self.lr)
        return [g_optimizer, d_optimizer], []
        # # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)  
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=200000)  
        # return [optimizer], [scheduler]