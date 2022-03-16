import pytorch_lightning as pl
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from transformers.optimization import get_cosine_schedule_with_warmup
from model.convnext_unet import ConvNextUnet
from model.gaussian_diffusion import GaussianDiffusion


class ClimateHackModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackModel, self).__init__()
        self.cfg = cfg
        self.model = ConvNextUnet(
            dim=64,
            out_dim=24,
            channels=36
        )
        self.diffusion = GaussianDiffusion(
            self.model, 
            image_size=128,
            channels=36,
            timesteps=100,
        )
        self.lr = self.cfg.trainer_cfg.lr

    def create_image(self, x):
        return self.diffusion.sample(x)

    def shared_step(self, batch, batch_idx):
        input_image, target_image = batch
        input_image, target_image = input_image/1024, target_image/1024
        loss = self.diffusion(target_image, input_image)
        #loss = loss_l1 + loss_ssim + loss_perceptual
        return dict(
            loss=loss,
            logs=dict(
                loss=loss.detach(),
            #     loss_l1=loss_l1.detach(),
            #     loss_ssim=loss_ssim.detach(),
            #     loss_perceptual=loss_perceptual.detach(),
            )
        )

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        # self.log_dict(outputs["logs"], prog_bar=True, sync_dist=True)
        if batch_idx % 200 == 0 and self.trainer.is_global_zero:
            input_image, target_image = batch
            input_image, target_image = input_image.float(), target_image.float()
            output_image = self.create_image(input_image)
            output_image=(output_image.view(-1, 1, 128, 128)).detach()
            target_image=(target_image.view(-1, 1, 128, 128)/1024).detach()
            save_image(output_image, "train_output_image.png")
            save_image(target_image, "train_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["train/output_image"].upload("train_output_image.png")
                self.logger.experiment["train/target_image"].upload("train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        # outputs = self.shared_step(batch, batch_idx)
        # outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        if batch_idx == 0 and self.trainer.is_global_zero:
            input_image, target_image = batch
            input_image, target_image = input_image.float(), target_image.float()
            output_image = self.create_image(input_image)
            output_image=(output_image.view(-1, 1, 128, 128)).detach()
            target_image=(target_image.view(-1, 1, 128, 128)/1024).detach()
            save_image(output_image, "val_output_image.png")
            save_image(target_image, "val_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["val/output_image"].upload("val_output_image.png")
                self.logger.experiment["val/target_image"].upload("val_target_image.png")

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)  
        return optimizer
        # g_optimizer = MADGRAD(self.model.parameters(), lr=self.lr)
        # d_optimizer = MADGRAD(self.discriminator.parameters(), lr=self.lr)
        # return [g_optimizer, d_optimizer], []
        # # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)  
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=200000)  
        # return [optimizer], [scheduler]