import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_optimizer import MADGRAD
from model.vqvae import VQVAE, VQVAE_Condition
from transformers.optimization import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy


class ClimateHackVqvaeConditionModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClimateHackVqvaeConditionModel, self).__init__()
        self.cfg = cfg
        self.vq = VQVAE(
            in_channel=1
        )
        self.vq_condition = VQVAE_Condition(
            in_channel=1
        )
        self.criterion_ce = nn.CrossEntropyLoss()
        self.train_acc_t = Accuracy()
        self.train_acc_b = Accuracy()
        self.val_acc_t = Accuracy()
        self.val_acc_b = Accuracy()
        self.lr = self.cfg.trainer_cfg.lr

    def forward(self, img):
        img = img / 1024
        t, b = self.vq_condition(img)
        t = t.permute(0, 4, 1, 2, 3)
        b = b.permute(0, 4, 1, 2, 3)
        return t, b

    def shared_step(self, batch, batch_idx):
        input, target = batch
        img = torch.cat([input, target], dim=1)
        img = img.float().unsqueeze(1)

        _, _, _, id_t, id_b = self.vq.encode(img/1024)
        pred_t, pred_b = self.forward(input.float().unsqueeze(1))

        loss_t = self.criterion_ce(pred_t, id_t[:, :3])
        loss_b = self.criterion_ce(pred_b, id_b[:, :6])
        loss = loss_t + loss_b

        return dict(
            loss=loss,
            id_t=id_t[:, :3].detach(),
            id_b=id_b[:, :6].detach(),
            id_t_pred=pred_t.detach(),
            id_b_pred=pred_b.detach(),
            logs=dict(
                loss=loss.detach(),
                loss_t=loss_t.detach(),
                loss_b=loss_b.detach(),
            ),
        )

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.train_acc_t(outputs["id_t_pred"], outputs["id_t"])
        self.train_acc_b(outputs["id_b_pred"], outputs["id_b"])
        outputs["logs"]["train_acc_t"] = self.train_acc_t
        outputs["logs"]["train_acc_b"] = self.train_acc_b
        self.log_dict(outputs["logs"], prog_bar=True)
        # if batch_idx % 100 == 0 and self.trainer.is_global_zero:
        #     save_image(outputs["images"]["output_image"], "train_output_image.png")
        #     save_image(outputs["images"]["target_image"], "train_target_image.png")
        #     if self.cfg.neptune_cfg.use_neptune:
        #         self.logger.experiment["train/output_image"].upload("train_output_image.png")
        #         self.logger.experiment["train/target_image"].upload("train_target_image.png")
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        outputs["logs"] = {"val_"+k: v for k, v in outputs["logs"].items()}
        self.val_acc_t(outputs["id_t_pred"], outputs["id_t"])
        self.val_acc_b(outputs["id_b_pred"], outputs["id_b"])
        outputs["logs"]["val_acc_t"] = self.val_acc_t
        outputs["logs"]["val_acc_b"] = self.val_acc_b
        self.log_dict(outputs["logs"], prog_bar=True)
        # if batch_idx == 0 and self.trainer.is_global_zero:
        #     save_image(outputs["images"]["output_image"], f"val_output_image_{self.num_val}.png")
        #     save_image(outputs["images"]["target_image"], f"val_target_image_{self.num_val}.png")
        #     if self.cfg.neptune_cfg.use_neptune:
        #         self.logger.experiment["val/output_image"].upload(f"val_output_image_{self.num_val}.png")
        #         self.logger.experiment["val/target_image"].upload(f"val_target_image_{self.num_val}.png")

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)  
        # return optimizer
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=200000)  
        return [optimizer], [scheduler]