import lightning.pytorch as pl
import torch
import torch.utils.data
from Components import DataComponents
from Components import Metrics
import torch.utils.tensorboard
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import subprocess
import threading
import time
from Networks import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class PLModule(pl.LightningModule):
    def __init__(self, network_arch, enable_val, enable_mid_visual, mid_visual_image, instance_mode,
                 use_sparse_label_train, use_sparse_label_val, use_sparse_label_test, logging):
        super().__init__()
        self.model = network_arch
        self.enable_val = enable_val
        self.enable_mid_visual = enable_mid_visual
        self.mid_visual_image = mid_visual_image
        self.instance_mode = instance_mode
        self.use_sparse_label_train = use_sparse_label_train
        self.use_sparse_label_val = use_sparse_label_val
        self.use_sparse_label_test = use_sparse_label_test
        self.logging = logging
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []
        self.initial_lr = 0.0005
        self.p_loss_fn = Metrics.BinaryMetrics("focal")
        self.c_loss_fn = Metrics.BinaryMetrics("focal")
        if enable_mid_visual:
            self.mid_visual_tensor = DataComponents.path_to_tensor(self.mid_visual_image).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, image):
        return self.model(image)

    def _step(self, batch, sparse):
        if self.instance_mode:
            img, lab, contour = batch
            p_output, c_output = self.forward(img)

            p_loss, p_dice, p_sensitivity, p_specificity = self.p_loss_fn(p_output, lab, False)
            c_loss, c_dice, c_sensitivity, c_specificity = self.c_loss_fn(c_output, contour, False)
            loss = p_loss + c_loss
            return loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity
        else:
            img, lab = batch
            output = self.forward(img)
            loss, dice, sensitivity, specificity = self.p_loss_fn(output, lab, sparse)
            return loss, dice, sensitivity, specificity

    def training_step(self, batch, batch_idx):
        if self.instance_mode:
            loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_train)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.train_metrics.append(
                [loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        else:
            loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_train)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.train_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.instance_mode:
            loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_val)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.val_metrics.append([loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        else:
            loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_val)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.val_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        if self.instance_mode:
            loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity = self._step(batch, self.use_sparse_label_test)
            self.test_metrics.append([loss, p_dice, c_dice, p_sensitivity, p_specificity, c_sensitivity, c_specificity])
        else:
            loss, dice, sensitivity, specificity = self._step(batch, self.use_sparse_label_test)
            self.test_metrics.append([loss, dice, sensitivity, specificity])
        return {'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        fused = True if device == "cuda" else False
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, fused=fused)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.25, patience=10,
                                                               threshold_mode='rel',
                                                               cooldown=0, min_lr=0.00025, verbose=True)
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
            if self.instance_mode:
                self.logger.experiment.add_scalar(f"{prefix}/Total Loss", epoch_averages[0], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Predict Dice", epoch_averages[1], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Predict Dice", epoch_averages[2],
                                                  self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Sensitivity", epoch_averages[3], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Specificity", epoch_averages[4], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Sensitivity", epoch_averages[5],
                                                  self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Specificity", epoch_averages[6],
                                                  self.current_epoch)
            else:
                self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Dice", epoch_averages[1], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", epoch_averages[2], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Specificity", epoch_averages[3], self.current_epoch)
            metrics_list.clear()

    def on_validation_epoch_end(self):
        if self.logging:
            self.log_metrics("Val", self.val_metrics)

    def on_train_epoch_end(self):
        if self.logging:
            self.log_metrics("Train", self.train_metrics)
            if device == 'cuda':
                vram_data = torch.cuda.mem_get_info()
                #vram_usage = torch.cuda.max_memory_allocated()/(1024**2)
                vram_usage = (vram_data[1] - vram_data[0])/(1024**2)
                self.logger.experiment.add_scalar(f"Other/VRAM Usage (MB)", vram_usage, self.current_epoch)
                torch.cuda.reset_peak_memory_stats()
            if self.enable_mid_visual:
                if self.instance_mode:
                    with torch.inference_mode():
                        mid_visual_pixel, mid_visual_contour = self.forward(self.mid_visual_tensor)
                        mid_visual_pixel = mid_visual_pixel[:, :, 0:1, :, :].squeeze([0, 1])
                        mid_visual_contour = mid_visual_contour[:, :, 0:1, :, :].squeeze([0, 1])
                    self.logger.experiment.add_image(f'Model Output/Pixel', mid_visual_pixel, self.current_epoch)
                    self.logger.experiment.add_image(f'Model Output/Contour', mid_visual_contour, self.current_epoch)
                else:
                    with torch.inference_mode():
                        mid_visual_result = self.forward(self.mid_visual_tensor)
                        mid_visual_result = mid_visual_result[:, :, 0:1, :, :].squeeze([0, 1])
                    self.logger.experiment.add_image(f'Model Output', mid_visual_result, self.current_epoch)

    def on_test_epoch_end(self):
        if self.logging:
            self.log_metrics("Test", self.test_metrics)



if __name__ == "__main__":
    model = PLModule(Instance_General.UNet(base_channels=4, z_to_xy_ratio=1, depth=4, type='Basic'),
                     True, False, None, True,
                     False, False, False, None)
    train_dataset = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters.csv",
                                                8, 128, 64, True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    #val_dataset = DataComponents.ValDataset("Datasets/val", "Augmentation Parameters.csv")
    #val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1, logger=None,
                         accelerator="gpu", enable_checkpointing=False,  # gradient_clip_val=0.001,
                         precision="32", enable_progress_bar=True, num_sanity_val_steps=0)
    # print(subprocess.run("tensorboard --logdir='lightning_logs'", shell=True))
    start_time = time.time()
    trainer.fit(model,
                #val_dataloaders=val_loader,
                train_dataloaders=train_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    '''
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
'''