import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data
from Components import DataComponents
from Components import Metrics
import torch.utils.tensorboard
from pytorch_lightning.loggers import TensorBoardLogger
import subprocess
import threading
import time
from Networks import *

logger = TensorBoardLogger('lightning_logs', name='Run')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
# Procedure parameters
# If set to true, then the script will read weights from an existing network rather than start a new one.
READ_EXISTING_NETWORK = False
# The filename for the existing network to read.
EXISTING_NETWORK_NAME = "placeholder.pth"
# Allow the training of the network.
ENABLE_TRAINING = True
# Allow the load and use of validation dataset during the training process.
ENABLE_VAL = True
# Allow the load and use of test dataset after the training process.
ENABLE_TEST = False
# Save the model as a file
SAVE_MODEL = True
# The name which it would be saved as
SAVE_MODEL_NAME = "placeholder.pth"
# Use the model to predict the segmentation of predict dataset.
ENABLE_PREDICT = True
# The csv file containing the parameters for image augmentation.
CSV = "Augmentation Parameters.csv"

# Training parameters
# The path to where the images for training are.
PATH_TO_TRAIN_DATASET = "Datasets/train"
# The batch size used for training, larger often trains faster and give more accurate results, but require more VRAM.
TRAIN_BATCH_SIZE = 1
# The number of epochs which the network will be trained for, the larger it is, the longer the training takes.
TRAIN_EPOCHS = 100
# Increase the effective number of training samples in each epoch by this factor.
TRAIN_MULTIPLIER = 8
# Determine the size of the network, higher means more channels in each layer, and hence runs slower and more accurate.
NETWORK_SIZE = 6
# The learning rate in which the network will start with.
BASE_LEARNING_RATE = 0.002
# Reduce learning rate by half when this number of epochs has passed and there's no decrease in loss.
LR_DECREASE_PATIENCE = 10

# Validation parameters
PATH_TO_VAL_DATASET = "Datasets/val"

# Test parameters
PATH_TO_TEST_DATASET = "Datasets/test"

# Prediction parameters
# The path to where the image for prediction are.
PATH_TO_PREDICT_DATASET = 'Datasets/predict'
# The path where the network will export its prediction result to.
PATH_TO_RESULT = 'Datasets/result'
# The height and width of sub-images the prediction image will be cut to. Larger means more accurate segmentation, but takes more VRAM.
HW_SIZE = 128
# Same as above but is depth.
DEPTH_SIZE = 64
# The overlaps in height and width between each adjacent sub-pictures. Larger means more accurate segmentation, but takes more VRAM.
HW_OVERLAP = 16
# Same as above but is depth.
DEPTH_OVERLAP = 16

# The architecture of the network
NETWORK_ARCH = UNets.UNet(base_channels=NETWORK_SIZE, depth=3)

def start_tensorboard():
    subprocess.run("tensorboard --logdir='lightning_logs'", shell=True)



class PLModule(pl.LightningModule):

    def __init__(self, arch):
        super().__init__()
        self.model = arch
        self.learning_rate = BASE_LEARNING_RATE
        self.train_metrics, self.val_metrics = [], []

    def forward(self, image):
        return self.model(image)

    def _step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        Loss_Fn = Metrics.BinaryMetrics()
        loss, dice, sensitivity, specificity = Loss_Fn(y_hat, y)
        return loss, dice, sensitivity, specificity

    def training_step(self, batch, batch_idx):
        loss, dice, sensitivity, specificity = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, dice, sensitivity, specificity = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.val_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        result = self.forward(x)
        result = torch.where(result >= 0.5, 1, 0).to(torch.int8)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=LR_DECREASE_PATIENCE,
                                                               threshold=0.001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0.0005, verbose=True)
        metrics = "val_loss" if ENABLE_VAL else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": metrics},
        }

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            pass
        else:
            scheduler.step(metric)

    def log_metrics(self, prefix, metrics_list):
        if metrics_list:
            epoch_averages = torch.stack([torch.tensor(metrics) for metrics in metrics_list]).mean(dim=0)
            self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch + 1)
            self.logger.experiment.add_scalar(f"{prefix}/Dice", epoch_averages[1], self.current_epoch + 1)
            self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", epoch_averages[2], self.current_epoch + 1)
            self.logger.experiment.add_scalar(f"{prefix}/Specificity", epoch_averages[3], self.current_epoch + 1)
            vram_data = torch.cuda.mem_get_info()
            vram_usage = (vram_data[1] - vram_data[0]) / (1024 ** 2)
            self.logger.experiment.add_scalar(f"{prefix}/VRAM Usage (MB)", vram_usage, self.current_epoch + 1)
            metrics_list.clear()

    def on_validation_epoch_end(self):
        self.log_metrics("Val", self.val_metrics)
        torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self):
        self.log_metrics("Train", self.train_metrics)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    if ENABLE_TRAINING:
        train_dataset = DataComponents.Train_Dataset(PATH_TO_TRAIN_DATASET, CSV, TRAIN_MULTIPLIER)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                   shuffle=True, num_workers=0)
        if ENABLE_VAL:
            val_dataset = DataComponents.Val_Dataset(PATH_TO_VAL_DATASET, CSV)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                     num_workers=0)
        else:
            val_loader = None
        if ENABLE_TEST:
            test_dataset = DataComponents.Val_Dataset(PATH_TO_TEST_DATASET, CSV)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                      num_workers=0)
    if READ_EXISTING_NETWORK:
        model = PLModule(NETWORK_ARCH)
        model.load_state_dict(torch.load(EXISTING_NETWORK_NAME))
    else:
        model = PLModule(NETWORK_ARCH)
    if ENABLE_TRAINING:
        tensorboard_thread = threading.Thread(target=start_tensorboard)
        tensorboard_thread.daemon = True
        tensorboard_thread.start()
        trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, log_every_n_steps=1, logger=logger,# profiler="simple",
                             accelerator="gpu", enable_checkpointing=False,# gradient_clip_val=0.001,
                             precision=32, enable_progress_bar=True)
        #print(subprocess.run("tensorboard --logdir='lightning_logs'", shell=True))
        start_time = time.time()
        trainer.fit(model,
                    val_dataloaders=val_loader,
                    train_dataloaders=train_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
    if ENABLE_TEST:
        trainer.test(model,
                     dataloaders=test_loader)
    if SAVE_MODEL:
        torch.save(model.state_dict(), SAVE_MODEL_NAME)
    if ENABLE_PREDICT:
        trainer = pl.Trainer(precision=32, enable_progress_bar=True, logger=False, accelerator="gpu"
                             #profiler="simple"
                             )
        predict_dataset = DataComponents.Predict_Dataset(PATH_TO_PREDICT_DATASET,
                                                         hw_size=HW_SIZE, depth_size=DEPTH_SIZE,
                                                         hw_overlap=HW_OVERLAP, depth_overlap=DEPTH_OVERLAP)
        predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
        # predictions = 一个list，包含了维度是(b,深度,高度,宽度)的输出张量(没有C！)
        original_volume = predict_dataset.__getoriginalvol__()
        start_time = time.time()
        predictions = trainer.predict(model, predict_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        DataComponents.predictions_to_final_img(predictions, direc=PATH_TO_RESULT, original_volume=original_volume,
                                                hw_size=HW_SIZE, depth_size=DEPTH_SIZE,
                                                hw_overlap=HW_OVERLAP, depth_overlap=DEPTH_OVERLAP)
