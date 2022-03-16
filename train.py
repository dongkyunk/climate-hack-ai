import sys
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset.climate_hack_datamodule import ClimateHackDatamodule
from model.climate_hack_model import ClimateHackModel
from config import register_configs, Config


@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    pl.seed_everything(cfg.trainer_cfg.seed)
    rank_zero_info(OmegaConf.to_yaml(cfg=cfg, resolve=True))
    pd.options.mode.chained_assignment = None 
    
    datamodule = ClimateHackDatamodule(cfg=cfg)
    model = ClimateHackModel(cfg=cfg)
    # model = ClimateHackModel.load_from_checkpoint(
    #     '/home/dongkyun/Desktop/Other/climate-hack-ai/save/two step unet/.neptune/two step unet/CLIM-52/checkpoints/epoch=18-val_loss=3.0656.ckpt', cfg=cfg
    # )

    # torch.save(model.model.state_dict(), 'model4.pth')
    # torch.save(model.refine.state_dict(), 'refine.pth')
    # # model = ClimateHackUnetModel(cfg=cfg)
    # #.load_from_checkpoint(
    # #     '/home/dongkyun/Desktop/Other/climate-hack-ai/save/unet_model/.neptune/unet_model/CLIM-42/checkpoints/epoch=19-val_loss=3.1292.ckpt', cfg=cfg
    # # )
    # #.load_from_checkpoint('/home/dongkyun/Desktop/Other/climate-hack-ai/save/vqvae_pretrain_new/.neptune/vqvae_pretrain_new/CLIM-21/checkpoints/epoch=19-val_loss=1.8061.ckpt', cfg=cfg)
    # # torch.save(model.model.state_dict(), 'unet.pth')
    # # torch.save(model.pixel_snail.state_dict(), 'pixel_snail.pth')
    # # model = ClimateHackModel.load_from_checkpoint('/home/dongkyun/Desktop/Other/climate-hack-ai/save/vqvae_pixelsnail_generator/.neptune/vqvae_pixelsnail_generator/CLIM-33/checkpoints/epoch=01-val_loss=0.4476.ckpt', cfg=cfg)
    # model.model.load_state_dict(torch.load('model.pth'))
    # # # # for param in model.top.top.parameters():
    # # # #     param.requires_grad = False
    # # # model.vq.load_state_dict(torch.load('/home/dongkyun/Desktop/Other/climate-hack-ai/save/vqvae_pretrain_new/vq.pth'))
    # # # for param in model.vq.parameters():
    # # #     param.requires_grad = False

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,
        monitor='val_loss',
    )

    trainer_args = dict(
        gpus=cfg.trainer_cfg.gpus,
        val_check_interval=cfg.trainer_cfg.val_check_interval,
        num_sanity_val_steps=0,#-1,
        max_epochs=cfg.trainer_cfg.epoch,
        callbacks=[checkpoint_callback],
        strategy=DDPPlugin(find_unused_parameters=True),
        profiler="simple",
        precision=16,
    )    
    if cfg.neptune_cfg.use_neptune:
        logger = NeptuneLogger(
            project=cfg.neptune_cfg.project_name,
            name=cfg.neptune_cfg.exp_name,
            tags=list(cfg.neptune_cfg.tags),
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWY4YTFhZS00NGU5LTQxOTUtOGI5NC04ZjgwOTJkMDFmNjYifQ==',
            log_model_checkpoints=False
        )
        logger.log_hyperparams(params=cfg.trainer_cfg.__dict__)
        trainer_args['logger'] = logger

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    sys.argv.append(f'hydra.run.dir={Config.path_cfg.save_dir}')
    register_configs()
    train()