import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchvision.utils import save_image
from torch_optimizer import MADGRAD
from model.vqvae import VQVAE
from model.pixel_snail import BottomPrior, TopPrior
from model.transformer import ImageTransformer
from transformers.optimization import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy


class ClimateHackModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackModel, self).__init__()
        self.cfg = cfg
        
        self.vq = VQVAE(1)
        self.top = TopPrior(512, 64, 2, 2, 2, 16, 128)  
        self.bottom = BottomPrior(512, 64, 2, 2, 2, 16, 128)
        # self.top = ImageTransformer()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_acc_t = Accuracy()
        self.train_acc_b = Accuracy()
        self.val_acc_t = Accuracy()
        self.val_acc_b = Accuracy()

        self.lr = self.cfg.trainer_cfg.lr

    def create_img(self, id_t, id_b):
        id_t, id_b = F.one_hot(id_t, 512).permute(0, 4, 1, 2, 3).float(), F.one_hot(id_b, 512).permute(0, 4, 1, 2, 3).float()
        id_t_pred = self.top(id_t)
        id_t_pred = F.one_hot(torch.argmax(id_t_pred, 1), 512).permute(0, 4, 1, 2, 3).float()
        id_b_pred = self.bottom(id_t_pred, id_b)
        img = self.vq.decode_latent(torch.argmax(id_t_pred, 1), torch.argmax(id_b_pred, 1))
        img = img*1024
        return img

    def forward(self, id_t, id_b):
        id_t, id_b = F.one_hot(id_t, 512).permute(0, 4, 1, 2, 3).float(), F.one_hot(id_b, 512).permute(0, 4, 1, 2, 3).float()
        id_t_pred = self.top(id_t)
        id_b_pred = self.bottom(id_t, id_b)
        return id_t_pred, id_b_pred

    def shared_step(self, batch, batch_idx, create_img=False):
        input_image, target_image = batch
        img = torch.cat([input_image, target_image], dim=1)
        img = img.float().unsqueeze(1)

        _, _, _, id_t, id_b = self.vq.encode(img/1024)
        id_t, id_b = id_t.detach(), id_b.detach()
        
        id_t_pred, id_b_pred = self(id_t, id_b)
        
        loss_t = self.criterion_ce(id_t_pred, id_t)
        loss_b = self.criterion_ce(id_b_pred, id_b)

        loss = loss_t + loss_b

        outputs = dict(
            loss=loss,
            id_t=id_t.detach(),
            id_b=id_b.detach(),
            id_t_pred=id_t_pred.detach(),
            id_b_pred=id_b_pred.detach(),
            logs=dict(
                loss=loss.detach(),
                loss_t=loss_t.detach(),
                loss_b=loss_b.detach(),
            )
        )
        
        if create_img:
            output_image = self.create_img(id_t, id_b)
            outputs["images"] = dict(
                output_image=(output_image.view(-1, 1, 128, 128)/1024).detach(),
                target_image=(img.view(-1, 1, 128, 128)/1024).detach(),
            )
        
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.train_acc_t(outputs["id_t_pred"], outputs["id_t"])
        self.train_acc_b(outputs["id_b_pred"], outputs["id_b"])
        outputs["logs"]["train_acc_t"] = self.train_acc_t
        outputs["logs"]["train_acc_b"] = self.train_acc_b
        self.log_dict(outputs["logs"], prog_bar=True, sync_dist=True)
        if batch_idx % 50 == 0 and self.trainer.is_global_zero:
            outputs = self.shared_step(batch, batch_idx, create_img=True)
            save_image(outputs["images"]["output_image"], "train_output_image.png")
            save_image(outputs["images"]["target_image"], "train_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["train/output_image"].upload("train_output_image.png")
                self.logger.experiment["train/target_image"].upload("train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        self.val_acc_t(outputs["id_t_pred"], outputs["id_t"])
        self.val_acc_b(outputs["id_b_pred"], outputs["id_b"])
        outputs["logs"]["val_acc_t"] = self.val_acc_t
        outputs["logs"]["val_acc_b"] = self.val_acc_b
        self.log_dict(outputs["logs"], prog_bar=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            outputs = self.shared_step(batch, batch_idx, create_img=True)
            save_image(outputs["images"]["output_image"], "val_output_image.png")
            save_image(outputs["images"]["target_image"], "val_target_image.png")
            if self.cfg.neptune_cfg.use_neptune:
                self.logger.experiment["val/output_image"].upload("val_output_image.png")
                self.logger.experiment["val/target_image"].upload("val_target_image.png")

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)  
        # return optimizer
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)  
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=200000)  
        return [optimizer], [scheduler]