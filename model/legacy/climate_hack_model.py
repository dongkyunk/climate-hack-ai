import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from model.transformer import ImageGPT, ImageTransformer, CnnBaseline
from pytorch_msssim import MS_SSIM
from model.loss import VGGPerceptualLoss


class ClimateHackModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackModel, self).__init__()
        self.cfg = cfg
        self.model = ImageTransformer()
        self.criterion_ssim = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=24)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss()
        self.lr = self.cfg.trainer_cfg.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        input_image, target_image = batch
        input_image = input_image.float()
        target_image = target_image.float()
        output_image, skip_image = self(input_image)
        # output_image, skip_image = self(torch.cat([input_image, target_image], dim=1))

        loss_ssim = 1 - self.criterion_ssim(output_image, target_image)
        loss_perceptual = self.criterion_perceptual(output_image.view(-1, 1, 64, 64), target_image.view(-1, 1, 64, 64)) * 1e-4

        loss_skip = 1 - self.criterion_ssim(skip_image, target_image)
        loss_skip_perceptual = self.criterion_perceptual(skip_image.view(-1, 1, 64, 64), target_image.view(-1, 1, 64, 64)) * 1e-4
        # loss_l1 = self.criterion_l1(output_image, target_image) /1024

        loss = loss_ssim + loss_perceptual
        return dict(
            loss=loss,
            logs=dict(
                loss=loss.detach(),
                loss_ssim=loss_ssim.detach(),
                loss_perceptual=loss_perceptual.detach(),
                loss_skip = loss_skip.detach(),
                loss_skip_perceptual = loss_skip_perceptual.detach(),
            ),
            images=dict(
                output_image=(output_image.view(-1, 1, 64, 64)/1024).detach(),
                target_image=(target_image.view(-1, 1, 64, 64)/1024).detach(),
                input_image=(input_image.view(-1, 1, 128, 128)/1024).detach(),
                skip_image=(skip_image.view(-1, 1, 64, 64)/1024).detach(),
            )
        )

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.log_dict({
            "loss_ssim": outputs["logs"]["loss_ssim"],
            "loss_perceptual": outputs["logs"]["loss_perceptual"],
            "loss_skip": outputs["logs"]["loss_skip"], 
            "loss_skip_perceptual": outputs["logs"]["loss_skip_perceptual"],
        }, prog_bar=True, sync_dist=True)
        if batch_idx % 50 == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], "output_image.png")
            save_image(outputs["images"]["target_image"], "target_image.png")
            save_image(outputs["images"]["input_image"], "input_image.png")
            save_image(outputs["images"]["skip_image"], "skip_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.log_dict({
            "val_loss": outputs["logs"]["loss"],
            "val_loss_ssim": outputs["logs"]["loss_ssim"],
            "val_loss_perceptual": outputs["logs"]["loss_perceptual"],
            "val_loss_skip": outputs["logs"]["loss_skip"], 
            "val_loss_skip_perceptual": outputs["logs"]["loss_skip_perceptual"],
        }, prog_bar=True, sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_image(outputs["images"]["output_image"], "output_image.png")
            save_image(outputs["images"]["target_image"], "target_image.png")
            save_image(outputs["images"]["input_image"], "input_image.png")
            save_image(outputs["images"]["skip_image"], "skip_image.png")

    def configure_optimizers(self):
        optimizer = MADGRAD(self.model.parameters(), lr=self.lr)  
        return optimizer