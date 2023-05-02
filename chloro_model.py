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

import pytorch_lightning
from pytorch_lightning.core.lightning import LightningModule
import json
import sys

# Custom imports
from datasets import ChloroDataset
from models import UNet, CustomViTAutoEnc, CustomUNet, CustomAttentionUNet
# from monai.networks.nets import UNet, ViTAutoEnc, AttentionNet

class Net(LightningModule):
    def __init__(self, config, model):
        super().__init__()

        self.config = config

        # instantiate model
        self._model = model

        self.train_loss_function = nn.MSELoss()
        self.val_loss_function = nn.MSELoss()
        self.test_loss_function = nn.MSELoss()

        # track best metrics
        self.best_val_loss = 0
        self.best_val_epoch = 0

        self.predictions = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):


        # instantiate training
        self.train_ds = ChloroDataset(
            files_directory = 'ForecastingAlgae/data/trainnc_data/',
            timesteps_in = self.config['time_in'],
            timesteps_out = self.config['time_out']
        )

        # validation datasets
        self.val_ds = ChloroDataset(
            files_directory = 'ForecastingAlgae/data/validationnc_data/',
            timesteps_in = self.config['time_in'],
            timesteps_out = self.config['time_out']
        )

        self.test_ds = ChloroDataset(
            files_directory = 'ForecastingAlgae/data/testnc_data/',
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
        optimizer = torch.optim.Adam(self._model.parameters(), self.config['lr'])
        return optimizer
        # optimizer = torch.optim.Adam(self._model.parameters(), self.config['lr'])
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True),
        #         "monitor": "train_loss",
        #         "frequency": 2
        #     }
        # }


    def training_step(self, batch, batch_idx):
        t_in, t_out, t_info = batch["t_gt"], batch["t_forecast"], batch["t_info"]
        pred = self.forward(t_in)
        loss = self.train_loss_function(pred, t_out)
        tensorboard_logs = {"train_loss": loss.item()}
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
            "val_loss": mean_val_loss,
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
        return {"log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        t_in, t_out, t_info = batch["t_gt"], batch["t_forecast"], batch["t_info"]
        pred = self.forward(t_in)
        self.predictions.append(pred)
        loss = self.test_loss_function(pred, t_out)
        tensorboard_logs = {"test_loss": loss.item()}
        return {"test_loss": loss, "test_number": len(pred)}


if __name__ == "__main__":
    # initialise the LightningModule
    parser = argparse.ArgumentParser(
        description='Chlorophyll Model Argument Parser')
    parser.add_argument("-test", "--test", type=bool, help="Test existing checkpoint instead of training new model",
                        required=False)
    args = parser.parse_args()

    # initialise Lightning's trainer.
    tb_logger = pytorch_lightning.loggers.CSVLogger(save_dir="./logs")
    trainer = pytorch_lightning.Trainer(
          gpus=[0],
          min_epochs=1,
          max_epochs=1,
          logger=tb_logger,
          enable_checkpointing=True,
          num_sanity_val_steps=1,
          log_every_n_steps=1,
          strategy='ddp'
    )

    config = {
          "batch_size": 64,
          "time_in": 10,
          "time_out": 20,
          "lr": 1e-4
      }


      # model = UNet(in_channels = 10, out_channels = 20)

      # model = CustomViTAutoEnc(
      #                           in_channels=10,
      #                           patch_size=(16,16),
      #                           img_size=(128,128),
      #                           out_channels = 20
      #                         )

    model = CustomUNet(
                        spatial_dims=2,
                        in_channels=10,
                        out_channels=20,
                        channels=(16,32,64,128,256),
                        strides=(2,2,2,2,2)
                      )

      # model = CustomAttentionUNet(
      #                     spatial_dims=2,
      #                     in_channels=10,
      #                     out_channels=20,
      #                     channels=1,
      #                     strides=1
      #                 )

    net = Net(config, model)
    net.prepare_data()

      # set up loggers and checkpoints
      # tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir="./logs")



    if args.test:
      print("Testing existing checkpoint")
      


      pathToPretrainedCheckpoint = "/content/ForecastingAlgae/logs/default/version_0/checkpoints/CustomUNet_Exp01_epoch=3-step=319.ckpt"
      model = LightningModule.load_from_checkpoint(pathToPretrainedCheckpoint)
      #model = LightningModule.load_from_checkpoint(pathToPretrainedCheckpoint)
      print()
      
      checkpoint = torch.load(pathToPretrainedCheckpoint)
      state_dict = checkpoint['state_dict']
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
          name = k[7:] # remove module.
          new_state_dict[name] = v

      model.load_state_dict(new_state_dict)
      print("loaded checkpoint")
      test_loader = net.test_dataloader()
      #model = LightningModule.load_from_checkpoint(pathToPretrainedModel)
      assert test_loader
      print("here")
      trainer.test(model)
      #predictions = trainer.predict(model, dataloaders=test_loader)
      #print(predictions.shape)


    else:

      config = {
          "batch_size": 64,
          "time_in": 10,
          "time_out": 20,
          "lr": 1e-4
      }

      # model = UNet(in_channels = 10, out_channels = 20)

      # model = CustomViTAutoEnc(
      #                           in_channels=10,
      #                           patch_size=(16,16),
      #                           img_size=(128,128),
      #                           out_channels = 20
      #                         )

      model = CustomUNet(
                          spatial_dims=2,
                          in_channels=10,
                          out_channels=20,
                          channels=(16,32,64,128,256),
                          strides=(2,2,2,2,2)
                      )

      # model = CustomAttentionUNet(
      #                     spatial_dims=2,
      #                     in_channels=10,
      #                     out_channels=20,
      #                     channels=1,
      #                     strides=1
      #                 )

      net = Net(config, model)


      # set up loggers and checkpoints
      # tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir="./logs")

      # train
      #trainer.fit(net)

      print(f"train completed, best_metric: {net.best_val_loss:.4f} " f"at epoch {net.best_val_epoch}")
      trainer.test(net)
      #preds = trainer.predict(dataloaders=net.test_dataloader())
      #print(net.predictions)
      print(len(net.predictions))
      print(net.predictions[0].shape)


      #sys.stdout.write(json.dumps(net.predictions))



'''
%load_ext tensorboard
%tensorboard --logdir=$log_dir

In Jupyter session, you can use the above code in a cell to visualize TensorBoard logging.
'''