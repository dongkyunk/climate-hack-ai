import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_optimizer import MADGRAD
from model.vqvae import VQVAE
from torchvision.utils import save_image
from pytorch_msssim import MS_SSIM
from model.loss import VGGPerceptualLoss
from transformers.optimization import get_cosine_schedule_with_warmup


class ClimateHackVqvaeModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackVqvaeModel, self).__init__()
        self.cfg = cfg
        self.model = VQVAE(
            in_channel=1
        )
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ssim = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=36)
        self.criterion_perceptual = VGGPerceptualLoss()

        self.lr = self.cfg.trainer_cfg.lr
        self.num_val = 0

    def forward(self, img):
        img = img / 1024
        out, loss_latent = self.model(img)
        return out*1024, loss_latent

    def shared_step(self, batch, batch_idx):
        input, target = batch
        img = torch.cat([input, target], dim=1)
        img = img.float().unsqueeze(1)
        # img = input.float().unsqueeze(1)
        out, loss_latent = self(img)
        loss_latent = loss_latent.mean() * 1e1
        loss_l1 = self.criterion_l1(out/1024, img/1024)*1e2
        loss_ssim = 1 - self.criterion_ssim(out.squeeze(), img.squeeze())*1e-1
        loss_perceptual = self.criterion_perceptual(out.view(-1, 1, 128, 128)/1024, img.view(-1, 1, 128, 128)/1024)*1e-1

        loss = loss_l1 + loss_latent + loss_ssim + loss_perceptual

        return dict(
            loss=loss,
            logs=dict(
                loss=loss.detach(),
                loss_l1=loss_l1.detach(),
                loss_latent=loss_latent.detach(),
                loss_ssim=loss_ssim.detach(),
                loss_perceptual=loss_perceptual.detach(),
            ),
            images=dict(
                output_image=(out.view(-1, 1, 128, 128)/1024).detach(),
                target_image=(img.view(-1, 1, 128, 128)/1024).detach(),
            )
        )

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.log_dict(outputs["logs"], prog_bar=True)
        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], "train_output_image.png")
            save_image(outputs["images"]["target_image"], "train_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["train/output_image"].upload("train_output_image.png")
                self.logger.experiment["train/target_image"].upload("train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        self.log_dict(outputs["logs"], prog_bar=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], f"val_output_image_{self.num_val}.png")
            save_image(outputs["images"]["target_image"], f"val_target_image_{self.num_val}.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["val/output_image"].upload(f"val_output_image_{self.num_val}.png")
                self.logger.experiment["val/target_image"].upload(f"val_target_image_{self.num_val}.png")
    
    def validation_epoch_end(self, outputs):
        self.num_val += 1

    def configure_optimizers(self):
        optimizer = MADGRAD(self.model.parameters(), lr=self.lr)  
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)  
        return [optimizer], [scheduler]