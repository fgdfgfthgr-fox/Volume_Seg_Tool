import lightning.pytorch as pl
import torch
import torch.utils.data

from Components import DataComponents
from Components import Metrics
from Components.AdEMAMix import AdEMAMix
import torch.utils.tensorboard
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
import time
from Networks import *
from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


device = "cuda" if torch.cuda.is_available() else "cpu"


class PLModule(pl.LightningModule):
    def __init__(self, network_arch, enable_val, enable_mid_visual, mid_visual_image, instance_mode,
                 use_sparse_label_train, use_sparse_label_val, use_sparse_label_test, logging):
        super().__init__()
        self.save_hyperparameters()
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
        self.lr = 3e-6 # Not the actual LR since it's automatically computed
        self.p_loss_fn = Metrics.BinaryMetrics("focal")
        self.u_p_loss_fn = Metrics.BinaryMetrics("bce_no_dice")
        self.c_loss_fn = Metrics.BinaryMetrics("dice+bce")
        if enable_mid_visual:
            self.mid_visual_tensor = DataComponents.path_to_tensor(self.mid_visual_image).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, image, type):
        return self.model(image, type)

    def _step(self, batch, sparse):
        if self.instance_mode:
            img, lab, type, contour = batch
            # 0 = normal, 1 = unsupervised
            if type[0] == 0:
                p_output, c_output = self.forward(img, type)
                p_loss, p_i, p_u, p_tp, p_fn, p_tn, p_fp = self.p_loss_fn(p_output, lab, False)
                c_loss, c_i, c_u, c_tp, c_fn, c_tn, c_fp = self.c_loss_fn(c_output, contour, False)
                loss = p_loss + c_loss
                return loss, p_i, c_i, p_u, c_u, p_tp, c_tp, p_fn, c_fn, p_tn, c_tn, p_fp, c_fp
            else:
                output = self.forward(img, type)
                loss, _, _, _, _, _, _ = self.u_p_loss_fn(output, img, False)
                return loss, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan
        else:
            img, lab, type = batch
            output = self.forward(img, type)
            if type[0] == 0:
                # loss, intersection, union, tp, fn, tn, fp
                return self.p_loss_fn(output, lab, sparse)
            else:
                # loss, nan, nan, nan, nan, nan, nan
                return self.u_p_loss_fn(output, lab, False)

    def training_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, self.use_sparse_label_train)
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        else:
            result_tuple = self._step(batch, self.use_sparse_label_train)
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def validation_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, self.use_sparse_label_val)
            self.log("val_loss", result_tuple[0], logger=False)
            self.val_metrics.append(result_tuple)
        else:
            result_tuple = self._step(batch, self.use_sparse_label_val)
            self.log("val_loss", result_tuple[0], logger=False)
            self.val_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def test_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, self.use_sparse_label_test)
            self.test_metrics.append(result_tuple)
        else:
            result_tuple = self._step(batch, self.use_sparse_label_test)
            self.test_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return [torch.sigmoid(tensor).to(torch.float16) for tensor in self.forward(batch[0], batch[1])]  # fp32 is unnecessary

    def configure_optimizers(self):
        #fused = True if device == "cuda" else False
        optimizer = AdEMAMix(self.parameters(), lr=self.lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=15,
                                                               threshold_mode='rel',
                                                               cooldown=0, min_lr=0.00001)
        metrics = "val_loss" if self.enable_val else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": metrics, "interval": "epoch", "frequency": 1},
        }

    def log_metrics(self, prefix, metrics_list):
        if metrics_list:
            epoch_averages = torch.stack([torch.tensor(metrics) for metrics in metrics_list]).nanmean(dim=0)
            if self.instance_mode:
                # Since each patch have equal number of pixels, it's safe to use their average intersection and union
                p_dice = epoch_averages[1]/epoch_averages[3]
                c_dice = epoch_averages[2]/epoch_averages[4]
                # Sensitivity = tp/(tp+fn)
                p_sensitivity = epoch_averages[5]/(epoch_averages[5]+epoch_averages[7])
                c_sensitivity = epoch_averages[6]/(epoch_averages[6]+epoch_averages[8])
                # Specificity = tn/(tn+fp)
                p_specificity = epoch_averages[9]/(epoch_averages[9]+epoch_averages[11])
                c_specificity = epoch_averages[10]/(epoch_averages[10]+epoch_averages[12])
                self.logger.experiment.add_scalar(f"{prefix}/Total Loss", epoch_averages[0], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Predict Dice", p_dice, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Predict Dice", c_dice,
                                                  self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Sensitivity", p_sensitivity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Specificity", c_sensitivity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Sensitivity", p_specificity,
                                                  self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Specificity", c_specificity,
                                                  self.current_epoch)
                self.log(f"{prefix}_epoch_dice", p_dice+c_dice, logger=False)
            else:
                dice = epoch_averages[1]/epoch_averages[2]
                sensitivity = epoch_averages[3]/(epoch_averages[3]+epoch_averages[4])
                specificity = epoch_averages[5]/(epoch_averages[5]+epoch_averages[6])
                self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Dice", dice, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", sensitivity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Specificity", specificity, self.current_epoch)
                self.log(f"{prefix}_epoch_dice", dice, logger=False)
            metrics_list.clear()

    def on_validation_epoch_end(self):
        if self.logging:
            with torch.no_grad():
                self.log_metrics("Val", self.val_metrics)
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.enable_val:
            sch.step(self.trainer.callback_metrics["val_loss"])

    def on_train_epoch_end(self):
        if self.logging:
            with torch.no_grad():
                self.log_metrics("Train", self.train_metrics)
                if device == 'cuda':
                    vram_data = torch.cuda.mem_get_info()
                    #vram_usage = torch.cuda.max_memory_allocated()/(1024**2)
                    vram_usage = (vram_data[1] - vram_data[0])/(1024**2)
                    self.logger.experiment.add_scalar(f"Other/VRAM Usage (MB)", vram_usage, self.current_epoch)
                    torch.cuda.reset_peak_memory_stats()
                if self.enable_mid_visual:
                    if self.instance_mode:
                        mid_visual_pixel, mid_visual_contour = self.forward(self.mid_visual_tensor, [0,])
                        mid_visual_pixel = torch.sigmoid(mid_visual_pixel[:, :, 0:1, :, :].squeeze([0, 1]))
                        mid_visual_contour = torch.sigmoid(mid_visual_contour[:, :, 0:1, :, :].squeeze([0, 1]))
                        self.logger.experiment.add_image(f'Model Output/Pixel', mid_visual_pixel, self.current_epoch)
                        self.logger.experiment.add_image(f'Model Output/Contour', mid_visual_contour, self.current_epoch)
                    else:
                        mid_visual_result = self.forward(self.mid_visual_tensor, [0,])
                        mid_visual_result = torch.sigmoid(mid_visual_result[:, :, 0:1, :, :].squeeze([0, 1]))
                        self.logger.experiment.add_image(f'Model Output', mid_visual_result, self.current_epoch)
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and not self.enable_val:
            sch.step(self.trainer.callback_metrics["train_loss"])

    def on_test_epoch_end(self):
        if self.logging:
            self.log_metrics("Test", self.test_metrics)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms, logger=True)


if __name__ == "__main__":
    train_label_mean = DataComponents.path_to_tensor("Datasets/train/Labels_image.tif", label=True).to(torch.float32).mean()
    model = PLModule(Semantic_General.UNet(base_channels=16, z_to_xy_ratio=1, depth=4, type='Residual', se=True, unsupervised=True, label_mean=train_label_mean),
                     True, False, 'Datasets/mid_visualiser/Ts-4c_ref_patch.tif', False,
                     False, False, False, True)
    val_dataset = DataComponents.ValDataset("Datasets/val", 256, 32, False, "Augmentation Parameters.csv")
    predict_dataset = DataComponents.Predict_Dataset("Datasets/predict", 232, 24, 12, 4, True)
    train_dataset_pos = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters.csv",
                                                    64,
                                                    256, 32, False, False, 0,
                                                    0,
                                                    0, 'positive')
    train_dataset_neg = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters.csv",
                                                    64,
                                                    256, 32, False, False, 0,
                                                    0,
                                                    0, 'negative')
    unsupervised_train_dataset = DataComponents.UnsupervisedDataset("Datasets/unsupervised_train",
                                                                    "Augmentation Parameters.csv",
                                                                    128,
                                                                    256, 32)
    train_dataset = DataComponents.CollectedDataset(train_dataset_pos, train_dataset_neg, unsupervised_train_dataset)
    #train_dataset = DataComponents.CollectedDataset(train_dataset_pos, train_dataset_neg)
    sampler = DataComponents.CollectedSampler(train_dataset, 2, unsupervised_train_dataset)
    #sampler = DataComponents.CollectedSampler(train_dataset, 2)
    collate_fn = DataComponents.custom_collate
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2,
                                               collate_fn=collate_fn, sampler=sampler,
                                               num_workers=6, pin_memory=True, persistent_workers=True)
    meta_info = predict_dataset.__getmetainfo__()
    predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    #model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="", filename="{epoch}-{Val_epoch_dice:.2f}", mode="max", save_weights_only=True)
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="", filename="test", mode="max",
                                                    monitor="Val_epoch_dice", save_weights_only=True, enable_version_counter=False)
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, logger=TensorBoardLogger(f'lightning_logs', name='test'),
                         accelerator="gpu", enable_checkpointing=True, gradient_clip_val=0.3,
                         precision="32", enable_progress_bar=True, num_sanity_val_steps=0, callbacks=[model_checkpoint,
                                                                                                      FineTuneLearningRateFinder(min_lr=0.00001, max_lr=0.1, attr_name='initial_lr')])
    # print(subprocess.run("tensorboard --logdir='lightning_logs'", shell=True))
    start_time = time.time()
    trainer.fit(model,
                val_dataloaders=val_loader,
                train_dataloaders=train_loader)
    model = PLModule.load_from_checkpoint('test.ckpt')
    predictions = trainer.predict(model, predict_loader)
    #del predict_loader, predict_dataset
    DataComponents.predictions_to_final_img(predictions, meta_info, direc='Datasets/result',
                                            hw_size=232, depth_size=24,
                                            hw_overlap=12,
                                            depth_overlap=4,
                                            TTA_hw=True)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {elapsed_time} seconds")

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
