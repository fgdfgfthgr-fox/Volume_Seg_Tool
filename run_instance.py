import lightning.pytorch as pl
import torch
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


def start_tensorboard():
    subprocess.run("tensorboard --logdir='/mnt/7018F20D48B6C548/PycharmProjects/Deeplearning/CV/lightning_logs'", shell=True)


class PLModuleInstance(pl.LightningModule):

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
        img, lab, contour = batch
        p_output, c_output = self.forward(img)
        Loss_Fn = Metrics.BinaryMetrics(use_log_cosh=False, sparse_label=sparse)
        p_loss, p_dice, p_sensitivity, p_specificity = Loss_Fn(p_output, lab)
        c_loss, c_dice, c_sensitivity, c_specificity = Loss_Fn(c_output, contour)
        loss = p_loss + c_loss
        return loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity

    def training_step(self, batch, batch_idx):
        loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_train)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_metrics.append([loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_val)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.val_metrics.append([loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_test)
        self.test_metrics.append([loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        return {'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        result_p, result_c = self.forward(x)
        result_p = torch.where(result_p >= 0.5, 1, 0).to(torch.int8)
        #result_c = torch.where(result_c >= 0.5, 1, 0).to(torch.int8)
        return result_p, result_c

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
            self.logger.experiment.add_scalar(f"{prefix}/Total Loss", epoch_averages[0], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Pixel Predict Dice", epoch_averages[1], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Contour Predict Dice", epoch_averages[2], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Pixel Sensitivity", epoch_averages[3], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Pixel Specificity", epoch_averages[4], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Contour Sensitivity", epoch_averages[5], self.current_epoch)
            self.logger.experiment.add_scalar(f"{prefix}/Contour Specificity", epoch_averages[6], self.current_epoch)
            metrics_list.clear()

    def on_validation_epoch_end(self):
        self.log_metrics("Val", self.val_metrics)

    def on_train_epoch_end(self):
        self.log_metrics("Train", self.train_metrics)

        lr = self.lr_schedulers().optimizer.param_groups[0]['lr']
        self.logger.experiment.add_scalar(f"Other/Learn Rate", lr, self.current_epoch)

        #vram_data = torch.cuda.mem_get_info()
        vram_usage = torch.cuda.max_memory_allocated()/(1024**2)
        #vram_usage = (vram_data[1]-vram_data[0])/(1024**2)
        self.logger.experiment.add_scalar(f"Other/VRAM Usage (MB)", vram_usage, self.current_epoch)
        torch.cuda.reset_peak_memory_stats()
        if self.enable_mid_visual:
            mid_visual_tensor = DataComponents.path_to_tensor(self.mid_visual_image).unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                mid_visual_pixel, mid_visual_contour = self.forward(mid_visual_tensor)
                mid_visual_pixel = mid_visual_pixel[:, :, 0:1, :, :].squeeze([0, 1])
                mid_visual_contour = mid_visual_contour[:, :, 0:1, :, :].squeeze([0, 1])
            self.logger.experiment.add_image(f'Model Output/Pixel', mid_visual_pixel, self.current_epoch)
            self.logger.experiment.add_image(f'Model Output/Contour', mid_visual_contour, self.current_epoch)

    def on_test_epoch_end(self):
        self.log_metrics("Test", self.test_metrics)


if __name__ == "__main__":
    model = PLModuleInstance(Instance_General.Residual(base_channels=16, z_to_xy_ratio=4, depth=5), 0.001, 6, 0.0005, True,
                             True, 'Datasets/mid_visualiser/image.tif',
                             False, False, False)
    #model.load_state_dict(torch.load('placeholder.pth'))
    tensorboard_thread = threading.Thread(target=start_tensorboard)
    tensorboard_thread.daemon = True
    tensorboard_thread.start()
    train_dataset = DataComponents.TrainDatasetInstance("Datasets/train", "Augmentation Parameters.csv", train_multiplier=32)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    val_dataset = DataComponents.ValDatasetInstance("Datasets/val", "Augmentation Parameters.csv")
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    trainer = pl.Trainer(max_epochs=200, log_every_n_steps=1, logger=logger,
                         accelerator="gpu", enable_checkpointing=False,  # gradient_clip_val=0.001,
                         precision="32", enable_progress_bar=True, num_sanity_val_steps=0)
    # print(subprocess.run("tensorboard --logdir='lightning_logs'", shell=True))
    start_time = time.time()
    trainer.fit(model,
                val_dataloaders=val_loader,
                train_dataloaders=train_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    torch.save(model.state_dict(), 'placeholder.pth')
    trainer = pl.Trainer(precision="32", enable_progress_bar=True, logger=False, accelerator="gpu")
    predict_dataset = DataComponents.Predict_Dataset('Datasets/predict',
                                                     hw_size=256, depth_size=64,
                                                     hw_overlap=32, depth_overlap=8)
    predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
    meta_info = predict_dataset.__getmetainfo__()
    start_time = time.time()
    predictions = trainer.predict(model, predict_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    DataComponents.predictions_to_final_img_instance(predictions, meta_info, direc='Datasets/result',
                                                     hw_size=256, depth_size=64,
                                                     hw_overlap=32, depth_overlap=8)
