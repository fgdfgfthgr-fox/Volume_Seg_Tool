import lightning.pytorch as pl
import torch
import torch.utils.data
import os
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
'''
# Settings
# Procedure parameters
# If set to true, then the script will read weights from an existing network rather than start a new one.
READ_EXISTING_NETWORK = True
# The filename for the existing network to read.
EXISTING_NETWORK_NAME = "placeholder.pth"
# Allow the training of the network.
ENABLE_TRAINING = False
# Allow the load and use of validation dataset during the training process.
ENABLE_VAL = True
# Allow the load and use of test dataset after the training process.
ENABLE_TEST = False
# Save the model as a file
SAVE_MODEL = False
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
TRAIN_EPOCHS = 50
# Determine the size of the network, higher means more channels in each layer, and hence runs slower and more accurate.
NETWORK_SIZE = 8
# The learning rate in which the network will start with.
BASE_LEARNING_RATE = 0.05
# Reduce learning rate by half when this number of epochs has passed and there's no decrease in loss.
LR_DECREASE_PATIENCE = 6

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
DEPTH_SIZE = 128
# The overlaps in height and width between each adjacent sub-pictures. Larger means more accurate segmentation, but takes more VRAM.
HW_OVERLAP = 16
# Similar to above.
DEPTH_OVERLAP = 16

# The architecture of the network
NETWORK_ARCH = Iterations_New.Initial(base_channels=NETWORK_SIZE)
'''

def start_tensorboard():
    subprocess.run("tensorboard --logdir='lightning_logs'", shell=True)


class PLModuleSemantic(pl.LightningModule):

    def __init__(self, network_arch, initial_lr, patience, min_lr, enable_val, enable_mid_visual, mid_visual_image,
                 use_sparse_label_train, use_sparse_label_val, use_sparse_label_test):
        super().__init__()
        self.model = network_arch
        self.initial_lr = initial_lr
        self.patience = patience
        self.min_lr = min_lr
        self.enable_val = enable_val
        self.enable_mid_visual = enable_mid_visual
        self.mid_visual_image = mid_visual_image
        self.use_sparse_label_train = use_sparse_label_train
        self.use_sparse_label_val = use_sparse_label_val
        self.use_sparse_label_test = use_sparse_label_test
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []

    def forward(self, image):
        return self.model(image)

    def _step(self, batch, sparse):
        x, y = batch
        y_hat = self.forward(x)
        Loss_Fn = Metrics.BinaryMetrics(use_log_cosh=False, sparse_label=sparse)
        loss, dice, sensitivity, specificity = Loss_Fn(y_hat, y)
        return loss, dice, sensitivity, specificity

    def training_step(self, batch, batch_idx):
        loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_train)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_val)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.val_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_test)
        self.test_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        result = self.forward(x)
        result = torch.where(result >= 0.5, 1, 0).to(torch.int8)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=self.patience,
                                                               threshold=0.001, threshold_mode='rel',
                                                               cooldown=0, min_lr=self.min_lr)
        metrics = "val_loss" if self.enable_val else "train_loss"
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
            self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Dice", epoch_averages[1], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", epoch_averages[2], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Specificity", epoch_averages[3], self.current_epoch)
            metrics_list.clear()

    def on_validation_epoch_end(self):
        self.log_metrics("Val", self.val_metrics)

    def on_train_epoch_end(self):
        self.log_metrics("Train", self.train_metrics)

        lr = self.lr_schedulers().optimizer.param_groups[0]['lr']
        self.logger.experiment.add_scalar(f"Other/Learn Rate", lr, self.current_epoch)

        #vram_data = torch.cuda.mem_get_info()
        vram_usage = torch.cuda.max_memory_allocated()/(1024**2)
        self.logger.experiment.add_scalar(f"Other/VRAM Usage (MB)", vram_usage, self.current_epoch)
        torch.cuda.reset_peak_memory_stats()
        if self.enable_mid_visual:
            mid_visual_tensor = DataComponents.path_to_tensor(self.mid_visual_image).unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                mid_visual_result = self.forward(mid_visual_tensor)
                mid_visual_result = mid_visual_result[:, :, 0:1, :, :].squeeze([0, 1])
            self.logger.experiment.add_image(f'Model Output', mid_visual_result, self.current_epoch)

    def on_test_epoch_end(self):
        self.log_metrics("Test", self.test_metrics)

'''
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    if ENABLE_TRAINING:
        train_dataset = DataComponents.TrainDataset(PATH_TO_TRAIN_DATASET, CSV, train_multiplier=4)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                   shuffle=True)
        if ENABLE_VAL:
            val_dataset = DataComponents.ValDataset(PATH_TO_VAL_DATASET, CSV)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=TRAIN_BATCH_SIZE)
        else:
            val_loader = None
        if ENABLE_TEST:
            test_dataset = DataComponents.ValDataset(PATH_TO_TEST_DATASET, CSV)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                      num_workers=0)
    if READ_EXISTING_NETWORK:
        model = PLModuleSemantic(NETWORK_ARCH, BASE_LEARNING_RATE, LR_DECREASE_PATIENCE, 0.001, ENABLE_VAL)
        model.load_state_dict(torch.load(EXISTING_NETWORK_NAME))
    else:
        model = PLModuleSemantic(NETWORK_ARCH, BASE_LEARNING_RATE, LR_DECREASE_PATIENCE, 0.001, ENABLE_VAL)
    if ENABLE_TRAINING:
        tensorboard_thread = threading.Thread(target=start_tensorboard)
        tensorboard_thread.daemon = True
        tensorboard_thread.start()
        trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, log_every_n_steps=1, logger=logger,
                             accelerator="gpu", enable_checkpointing=False,# gradient_clip_val=0.001,
                             precision="32", enable_progress_bar=True, num_sanity_val_steps=0)
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
        trainer = pl.Trainer(precision="32", enable_progress_bar=True, logger=False, accelerator="gpu"
                             #profiler="simple"
                             )
        predict_dataset = DataComponents.Predict_Dataset(PATH_TO_PREDICT_DATASET,
                                                         hw_size=HW_SIZE, depth_size=DEPTH_SIZE,
                                                         hw_overlap=HW_OVERLAP, depth_overlap=DEPTH_OVERLAP)
        predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
        # predictions = 一个list，包含了维度是(b,深度,高度,宽度)的输出张量(没有C！)
        meta_info = predict_dataset.__getmetainfo__()
        start_time = time.time()
        predictions = trainer.predict(model, predict_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        DataComponents.predictions_to_final_img(predictions, meta_info, direc=PATH_TO_RESULT,
                                                hw_size=HW_SIZE, depth_size=DEPTH_SIZE,
                                                hw_overlap=HW_OVERLAP, depth_overlap=DEPTH_OVERLAP)
'''
