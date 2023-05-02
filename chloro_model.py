'''
This file contains a generalized PyTorch Lightning configuration for training a 
3D time-dependent DL model (Conv3D backbone, UNET baseline) for forward-forecasting
Chlorophyll concentration from gathered NASA dataset.

Of note are files in datasets/ and models/.

Experiment tracking is performed in TensorBoard.
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from collections import OrderedDict

# pytorch lightning imports
import pytorch_lightning # parent import
from pytorch_lightning.core.lightning import LightningModule # gen lightning module
from pytorch_lightning.callbacks import RichProgressBar # rich output to stdout
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping # stopping model training early if satisfactory
from pytorch_lightning.callbacks import ModelCheckpoint # logging model checkpoints
from pytorch_lightning.callbacks import LearningRateMonitor # logging LR
from pytorch_lightning.loggers import CSVLogger # logging metrics
from pytorch_lightning.profiler import SimpleProfiler # for speed profiling

import json
import sys
import pickle
import numpy as np
# Custom imports
from datasets import ChloroDataset
from models import UNet, CustomViTAutoEnc, CustomUNet, CustomAttentionUNet
# from monai.networks.nets import UNet, ViTAutoEnc, AttentionNet

class Net(LightningModule):
    def _init_(self, config, model):
        super()._init_()

        self.config = config

        # instantiate model
        self._model = model
        self.learning_rate = self.config['lr']

        self.train_loss_function = nn.MSELoss()
        self.val_loss_function = nn.MSELoss()
        self.test_loss_function = nn.MSELoss()

        # track best metrics
        self.best_val_loss = 0
        self.best_val_epoch = 0


    def forward(self, x):
        return self._model(x)

    def prepare_data(self):


        # instantiate training
        self.train_ds = ChloroDataset(
            files_directory = './data/trainnc_data/',
            timesteps_in = self.config['time_in'],
            timesteps_out = self.config['time_out']
        )

        # validation datasets
        self.val_ds = ChloroDataset(
            files_directory = './data/validationnc_data/',
            timesteps_in = self.config['time_in'],
            timesteps_out = self.config['time_out']
        )

        self.test_ds = ChloroDataset(
            files_directory = './data/testnc_data/',
            timesteps_in = self.config['time_in'],
            timesteps_out = self.config['time_out']
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            # shuffle=True,
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=self.config['batch_size'], num_workers=4)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds, batch_size=self.config['batch_size'], num_workers=4)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        t_in, t_out, t_info = batch["t_gt"], batch["t_forecast"], batch["t_info"]
        pred = self.forward(t_in)
        loss = self.train_loss_function(pred, t_out)
        tensorboard_logs = {"train_loss": loss.item()}
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        t_in, t_out, t_info = batch["t_gt"], batch["t_forecast"], batch["t_info"]
        pred = self.forward(t_in)
        loss = self.val_loss_function(pred, t_out)
        tensorboard_logs = {"val_loss": loss.item()}
        return {"val_loss": loss, "val_number": len(pred)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "mean_val_loss": mean_val_loss,
        }
        if mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean loss: {mean_val_loss:.4f}"
            f"\nbest mean loss: {self.best_val_loss:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)
        return {"log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        t_in, t_out, t_info = batch["t_gt"], batch["t_forecast"], batch["t_info"]
        pred = self.forward(t_in)
        loss = self.test_loss_function(pred, t_out)
        tensorboard_logs = {"test_loss": loss.item()}
        print(f"current test loss @ {batch_idx}:", loss)
        self.log("test_loss", loss.item(), prog_bar=True, sync_dist=True)
        return {"test_loss": loss, "test_number": len(pred)}
    
    def test_epoch_end(self, outputs):
        test_loss, num_items = 0, 0
        for output in outputs:
            test_loss += output["test_loss"].sum().item()
            num_items += output["test_number"]
        mean_test_loss = torch.tensor(test_loss / num_items)
        tensorboard_logs = {
            "mean_test_loss": mean_test_loss,
        }
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean test loss: {mean_test_loss:.4f}"
        )
        self.log("test_loss", mean_test_loss, prog_bar=True, sync_dist=True)
        return {"log": tensorboard_logs}

if _name_ == "_main_":
    
    config = {
          "model": "UNET",
          "batch_size": 32,
          "time_in": 10,
          "time_out": 20,
          "lr": 1e-4
      }
    
    model_name = config['model']
    # initialise Lightning's trainer.
    tb_logger = pytorch_lightning.loggers.CSVLogger(save_dir=f"./logs/{model_name}/")

#     model = CustomViTAutoEnc(
#                             in_channels=10,
#                             patch_size=(16,16),
#                             img_size=(128,128),
#                             out_channels = 20,
#                             dropout_rate = 0.4
#                             )

    model = CustomUNet(
                        spatial_dims=2,
                        in_channels=10,
                        out_channels=20,
                        channels=(16,32,64,128,256),
                        strides=(2,2,2,2,2),
                        dropout = 0.4
                      )

#     model = CustomAttentionUNet(
#                       spatial_dims=2,
#                       in_channels=10,
#                       out_channels=20,
#                       channels=(16,32,64,128,256),
#                       strides=(2,2,2,2,2),
#                       dropout = 0.4
#     )
        
    net = Net(config, model)
    
        # Init Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor = "val_loss",
        mode = "min",
        dirpath = f"./checkpoints/{config['model']}/"
    )

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        min_epochs=1,
        max_epochs=4,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        strategy='ddp',
        callbacks = [checkpoint_callback]
    )

    # train
    trainer.fit(net)
    print(f"train completed, best_metric: {net.best_val_loss:.4f} " f"at epoch {net.best_val_epoch}")

    # test
    trainer.test(net)