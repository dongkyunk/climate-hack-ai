import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from transformers.optimization import get_cosine_schedule_with_warmup
from model.convnext_unet import ConvNextUnet
from model.convnext_encoder import ConvNextEncoder
from pytorch_msssim import MS_SSIM
from model.loss import VGGPerceptualLoss


class ClimateHackModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackModel, self).__init__()
        self.cfg = cfg
        # self.model = ConvNextUnet(
        #     dim=64,
        #     out_dim=24,
        #     channels=12,
        #     with_time_emb=False
        # )
        self.model = ConvNextEncoder()
        self.lr = self.cfg.trainer_cfg.lr
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ssim = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=24)
        self.criterion_perceptual = VGGPerceptualLoss()

    def loss_fn(self, outputs, labels):
        loss_l1 = self.criterion_l1(outputs, labels)*1e2
        # labels = labels.type_as(outputs)
        # loss_ssim = 1 - self.criterion_ssim(outputs.squeeze()*1024, labels.squeeze()*1024)*1e-1
        # loss_perceptual = self.criterion_perceptual(outputs.view(-1, 1, 128, 128), labels.view(-1, 1, 128, 128))*1e-1
        # loss_perceptual = self.criterion_perceptual(outputs.view(-1, 1, 64, 64), labels.view(-1, 1, 64, 64))*1e-1
        loss = loss_l1  #+ loss_perceptual
        return loss, loss_l1#, loss_perceptual

    def shared_step(self, batch, batch_idx):
        input_image, target_image = batch
        input_image, target_image = input_image/1024, target_image/1024
        target_image = target_image[:, :, 32:96, 32:96]

        # output_image, _ = self.model(input_image, 0)

        output_image = self.model(input_image)

        loss, loss_l1 = self.loss_fn(output_image, target_image)

        return dict(
            loss=loss,
            logs=dict(
                loss=loss.detach(),
                loss_l1=loss_l1.detach(),
                # loss_ssim=loss_ssim.detach(),
                # loss_perceptual=loss_perceptual.detach(),
            ),
            images=dict(
                output_image=output_image.view(-1, 1, 64, 64).detach(),
                target_image=target_image.view(-1, 1, 64, 64).detach(),
            )
        )

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.log_dict(outputs["logs"], prog_bar=True, sync_dist=True)
        if batch_idx % 200 == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"],
                       "train_output_image.png")
            save_image(outputs["images"]["target_image"],
                       "train_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["train/output_image"].upload(
                    "train_output_image.png")
                self.logger.experiment["train/target_image"].upload(
                    "train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        self.log_dict(outputs["logs"], prog_bar=True, sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"],
                       "val_output_image.png")
            save_image(outputs["images"]["target_image"],
                       "val_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["val/output_image"].upload(
                    "val_output_image.png")
                self.logger.experiment["val/target_image"].upload(
                    "val_target_image.png")

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)
        return optimizer
        # g_optimizer = MADGRAD(self.model.parameters(), lr=self.lr)
        # d_optimizer = MADGRAD(self.discriminator.parameters(), lr=self.lr)
        # return [g_optimizer, d_optimizer], []
        # # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=200000)
        # return [optimizer], [scheduler]
