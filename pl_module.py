import math

import lightning.pytorch as pl
import torch
import torch.utils.data
import time
import tracemalloc
import torch.utils.tensorboard

from Components import DataComponents
from Components import Metrics
from Components.AdEMAMix import AdEMAMix
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from Networks import *
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


device = "cuda" if torch.cuda.is_available() else "cpu"

def pick_arch(arch_args):
    arch, base_channels, depth, z_to_xy_ratio, se, label_mean, contour_mean = arch_args
    if arch == "HalfUNetBasic":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Basic', se, label_mean)
    elif arch == "HalfUNetGhost":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Ghost', se, label_mean)
    elif arch == "HalfUNetResidual":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Residual', se, label_mean)
    elif arch == "HalfUNetResidualBottleneck":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se, label_mean)
    elif arch == "UNetBasic":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'Basic', se, label_mean)
    elif arch == "UNetResidual_Recommended":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'Residual', se, label_mean)
    elif arch == "UNetResidualBottleneck":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se, label_mean)
    elif arch == "SegNet":
        return Semantic_SegNets.Auto(base_channels, depth, z_to_xy_ratio, se, label_mean)
    #elif arch == "Tiniest":
    #    return Testing_Models.Tiniest(base_channels, depth, z_to_xy_ratio)
    #elif arch == "SingleTopLayer":
    #    return Testing_Models.SingleTopLayer(base_channels, depth, z_to_xy_ratio, 'Basic', se)
    elif arch == "InstanceBasic":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'Basic', se, label_mean, contour_mean)
    elif arch == "InstanceResidual_Recommended":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'Residual', se, label_mean, contour_mean)
    elif arch == "InstanceResidualBottleneck":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se, label_mean, contour_mean)


def weight_sequence(alpha, n):

    # Generate a sequence of weights that decrease exponentially
    weights = [alpha ** i for i in range(n)]

    # Normalize the weights so that they sum to 1
    total_weight = sum(weights)
    sequence = [w / total_weight for w in weights]

    return sequence


class PLModule(pl.LightningModule):
    def __init__(self, arch_args, enable_val, enable_mid_visual, mid_visual_image, instance_mode,
                 use_sparse_label_train, use_sparse_label_val, use_sparse_label_test, logging):
        super().__init__()
        self.save_hyperparameters()
        self.network = pick_arch(arch_args)
        self.p_weights = weight_sequence(0.8, arch_args[2]-1)
        self.enable_val = enable_val
        self.enable_mid_visual = enable_mid_visual
        self.mid_visual_image = mid_visual_image
        self.instance_mode = instance_mode
        self.use_sparse_label_train = use_sparse_label_train
        self.use_sparse_label_val = use_sparse_label_val
        self.use_sparse_label_test = use_sparse_label_test
        self.logging = logging
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []
        self.lr = 1e-4 # Not the actual LR since it's automatically computed
        self.pixel_ramp_steps = 2048  # Ramp-up the weight of entropy minimisation during the initial 2048 steps
        self.unsupervised_weight = 0.1
        self.p_loss_fn = Metrics.BinaryMetrics("focal")
        self.c_loss_fn = Metrics.BinaryMetrics("dice+bce")
        if enable_mid_visual:
            self.mid_visual_tensor = torch.from_numpy(DataComponents.path_to_array(self.mid_visual_image)).unsqueeze(0).unsqueeze(0).to(device)

        self.dice_threshold_reached = False
        self.starting_step = None

    def forward(self, image):
        return self.network(image)

    def compute_ramp_up_weight(self, ramp_steps):
        # Check if the dice score threshold has been reached
        if not self.dice_threshold_reached:
            return 0.0  # No ramp-up until dice score exceeds threshold

        # Get the current global step and compute the ramp-up weight
        if self.starting_step is None:
            self.starting_step = self.global_step
        current_step = self.global_step - self.starting_step
        if current_step < ramp_steps:
            return math.e ** (-5 * ((1 - (current_step / ramp_steps)) ** 2)) * self.unsupervised_weight
        else:
            return self.unsupervised_weight

    @staticmethod
    def entropy_preprocess(value):
        return 0.999 * torch.sigmoid(value) + 5e-4

    def configure_optimizers(self):
        fused = True if device == "cuda" else False
        optimizer = AdEMAMix(self.parameters(), lr=self.lr)#, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=50,
                                                               threshold_mode='rel',
                                                               cooldown=0, min_lr=0.00001)
        metrics = "val_loss" if self.enable_val else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": metrics, "interval": "epoch", "frequency": 1},
        }

    def _step(self, batch, type, sparse):
        pixel_ramp_weight = self.compute_ramp_up_weight(self.pixel_ramp_steps)
        if self.instance_mode:
            if type == 'Supervised':
                img, lab, contour = batch
            else:
                img = batch[0]
            p_outputs, c_output = self.forward(img)
            p_outputs = [p_output * self.p_weights[i] for i, p_output in enumerate(p_outputs)]
            p_output = torch.sum(torch.stack(p_outputs), dim=0)
            if self.dice_threshold_reached:
                #sigmoid_c_output = self.entropy_preprocess(c_output)
                sigmoid_p_output = self.entropy_preprocess(p_output)
                entropy_loss_p = (-sigmoid_p_output * torch.log(sigmoid_p_output)).mean()
                #entropy_loss_p = sum([(-value * torch.log(value)).mean() for value in sigmoid_p_outputs]) / len(sigmoid_p_outputs)

                entropy = entropy_loss_p #+ entropy_loss_c
                entropy_loss = entropy * pixel_ramp_weight
            else:
                entropy = torch.tensor(0.0, requires_grad=False)
                entropy_loss = torch.tensor(0.0, requires_grad=False)
            if type == 'Supervised':
                p_loss, p_i, p_u, p_tp, p_fn, p_tn, p_fp = self.p_loss_fn(p_output, lab, False)
                c_loss, c_i, c_u, c_tp, c_fn, c_tn, c_fp = self.c_loss_fn(c_output, contour, False)
                supervised_loss = p_loss + c_loss
                loss = supervised_loss + entropy_loss

                return loss, p_i, c_i, p_u, c_u, p_tp, c_tp, p_fn, c_fn, p_tn, c_tn, p_fp, c_fp, entropy
            else:
                return entropy_loss, *(torch.nan,) * 12, entropy
        else:
            if type == 'Supervised':
                img, lab = batch
            else:
                img = batch[0]
            outputs = self.forward(img)
            outputs = [output * self.p_weights[i] for i, output in enumerate(outputs)]
            output = torch.sum(torch.stack(outputs), dim=0)
            if self.dice_threshold_reached:
                sigmoid_outputs = self.entropy_preprocess(output)
                entropy = (-sigmoid_outputs * torch.log(sigmoid_outputs)).mean()
                entropy_loss = entropy * pixel_ramp_weight
            else:
                entropy = torch.tensor(0.0, requires_grad=False)
                entropy_loss = torch.tensor(0.0, requires_grad=False)
            if type == 'Supervised':
                supervised_loss, i, u, tp, fn, tn, fp = self.p_loss_fn(output, lab, sparse)
                loss = supervised_loss + entropy_loss
                return loss, i, u, tp, fn, tn, fp, entropy
            else:
                # loss, nan, nan, nan, nan, nan, nan
                return entropy_loss, *(torch.nan,) * 6, entropy

    def training_step(self, batch, batch_idx):
        if self.instance_mode:
            if len(batch) == 3:
                type = 'Supervised'
            else:
                type = 'Unsupervised'
            if not self.dice_threshold_reached and type == 'Unsupervised':
                return None
            result_tuple = self._step(batch, type, False)
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        else:
            if len(batch) == 2:
                type = 'Supervised'
            else:
                type = 'Unsupervised'
            if not self.dice_threshold_reached and type == 'Unsupervised':
                return None
            result_tuple = self._step(batch, type, self.use_sparse_label_train)
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def validation_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, 'Supervised', False)
            self.log("val_loss", result_tuple[0], logger=False)
            self.val_metrics.append(result_tuple)
        else:
            result_tuple = self._step(batch, 'Supervised', self.use_sparse_label_val)
            self.log("val_loss", result_tuple[0], logger=False)
            self.val_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def test_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, 'Supervised', self.use_sparse_label_test)
            self.test_metrics.append(result_tuple)
        else:
            result_tuple = self._step(batch, 'Supervised', self.use_sparse_label_test)
            self.test_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        def apply_augmentation(data, i):
            if i % 2 == 0:
                # D
                data = torch.flip(data, [2])
            if i % 4 <= 1:
                # H
                data = torch.flip(data, [3])
            if i % 8 <= 3:
                # W
                data = torch.flip(data, [4])
            return data

        TTA_results = []
        for i in range(8):
            aug_batch = apply_augmentation(batch, i)
            outputs = self.forward(aug_batch)

            if isinstance(outputs, tuple):
                p_outputs = [p_output * self.p_weights[i] for i, p_output in enumerate(outputs[0])]
                p_outputs = torch.sigmoid(torch.sum(torch.stack(p_outputs), dim=0)).to(torch.float16)
                #p_outputs = torch.sigmoid(torch.mean(torch.stack(outputs[0], dim=0), dim=0)).to(torch.float16)
                c_outputs = torch.sigmoid(outputs[1]).to(torch.float16)
                p_outputs = apply_augmentation(p_outputs, i)
                c_outputs = apply_augmentation(c_outputs, i)
                TTA_results.append((p_outputs, c_outputs))
            else:
                p_outputs = [p_output * self.p_weights[i] for i, p_output in enumerate(outputs)]
                p_outputs = torch.sigmoid(torch.sum(torch.stack(p_outputs), dim=0)).to(torch.float16)
                #p_outputs = torch.sigmoid(torch.mean(torch.stack(outputs, dim=0), dim=0)).to(torch.float16)
                p_outputs = apply_augmentation(p_outputs, i)
                TTA_results.append(p_outputs)

        if isinstance(TTA_results[0], tuple):
            p, c = zip(*TTA_results)
            return torch.mean(torch.stack(p, dim=0), dim=0), torch.mean(torch.stack(c, dim=0), dim=0)
        else:
            return torch.mean(torch.stack(TTA_results, dim=0), dim=0)

    def log_metrics(self, prefix, metrics_list):
        if metrics_list:
            epoch_averages = torch.stack([torch.tensor(metrics) for metrics in metrics_list]).nanmean(dim=0)
            required_prefix = 'Val' if self.enable_val else 'Train'
            if self.instance_mode:
                # Since each patch have equal number of pixels, it's safe to use their average intersection and union
                p_dice = epoch_averages[1]/epoch_averages[3]
                c_dice = epoch_averages[2]/epoch_averages[4]
                # Check if Contour Predict Dice >= 0.6
                if c_dice >= 0.6 and self.dice_threshold_reached == False and prefix == required_prefix:
                    self.dice_threshold_reached = True
                    print('Starts working on Unsupervised Samples via entropy minimisation...')
                    print('\nIgnore this if you are not using unsupervised learning.')
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
                self.logger.experiment.add_scalar(f"{prefix}/Pixel Specificity", p_specificity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Sensitivity", c_sensitivity,
                                                  self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Contour Specificity", c_specificity,
                                                  self.current_epoch)
                if epoch_averages[13] != 0:
                    self.logger.experiment.add_scalar(f"{prefix}/Entropy", epoch_averages[13], self.current_epoch)
                self.log(f"{prefix}_epoch_dice", p_dice+c_dice, logger=False)
            else:
                dice = epoch_averages[1]/epoch_averages[2]
                # Check if Dice >= 0.85
                if ((dice >= 0.85 and required_prefix == 'Train') or (dice >= 0.8 and required_prefix == 'Val')) and required_prefix == prefix and self.dice_threshold_reached == False:
                    self.dice_threshold_reached = True
                    print('\nStarts working on Unsupervised Samples via entropy minimisation...\n')
                    print('\nIgnore this if you are not using unsupervised learning.\n')
                sensitivity = epoch_averages[3]/(epoch_averages[3]+epoch_averages[4])
                specificity = epoch_averages[5]/(epoch_averages[5]+epoch_averages[6])
                self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Dice", dice, self.current_epoch)
                if epoch_averages[7] != 0:
                    self.logger.experiment.add_scalar(f"{prefix}/Entropy", epoch_averages[7], self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", sensitivity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Specificity", specificity, self.current_epoch)
                self.log(f"{prefix}_epoch_dice", dice, logger=False)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            self.log_metrics("Val", self.val_metrics)
            self.val_metrics.clear()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau) and self.enable_val:
            self.lr_schedulers().step(self.trainer.callback_metrics["val_loss"])

    def on_train_epoch_end(self):
        with torch.no_grad():
            self.log_metrics("Train", self.train_metrics)
            self.train_metrics.clear()
            if device == 'cuda':
                vram_data = torch.cuda.mem_get_info()
                #vram_usage = torch.cuda.max_memory_allocated()/(1024**2)
                vram_usage = (vram_data[1] - vram_data[0])/(1024**2)
                self.logger.experiment.add_scalar(f"Other/VRAM Usage (MB)", vram_usage, self.current_epoch)
                torch.cuda.reset_peak_memory_stats()
            if self.enable_mid_visual:
                if self.instance_mode:
                    p_outputs, c_outputs = self.forward(self.mid_visual_tensor)
                    sigmoid_p_outputs, sigmoid_c_outputs = [self.entropy_preprocess(scale) for scale in p_outputs], self.entropy_preprocess(c_outputs)
                    mid_visual_pixel = [output * self.p_weights[i] for i, output in enumerate(sigmoid_p_outputs)]
                    mid_visual_pixel = torch.sum(torch.stack(mid_visual_pixel), dim=0)
                    mid_visual_pixel = mid_visual_pixel[:, :, 0:1, :, :].squeeze([0, 1])
                    mid_visual_contour = sigmoid_c_outputs[:, :, 0:1, :, :].squeeze([0, 1])
                    self.logger.experiment.add_image(f'Model Output/Pixel', mid_visual_pixel, self.current_epoch)
                    self.logger.experiment.add_image(f'Model Output/Contour', mid_visual_contour, self.current_epoch)
                    pass
                else:
                    outputs = self.forward(self.mid_visual_tensor)
                    mid_visual_result = [output * self.p_weights[i] for i, output in enumerate(outputs)]
                    mid_visual_result = torch.sum(torch.stack(mid_visual_result), dim=0)
                    mid_visual_result = torch.sigmoid(mid_visual_result[:, :, 0:1, :, :].squeeze([0, 1]))
                    self.logger.experiment.add_image(f'Model Output', mid_visual_result, self.current_epoch)
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and not self.enable_val:
            sch.step(self.trainer.callback_metrics["train_loss"])

    def on_test_epoch_end(self):
        if self.logging:
            self.log_metrics("Test", self.test_metrics)
            self.test_metrics.clear()

    '''def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.network, norm_type=2)
        self.log_dict(norms, logger=True)'''


if __name__ == "__main__":
    #tracemalloc.start()
    #snap1 = tracemalloc.take_snapshot()
    sizes = [(152, 44, 4), (192, 56, 5)]
    channel_counts = [4, 8, 16]
    precisions = ['bf16-mixed', '32']
    batch_sizes = [2]
    for size in sizes:
        for channel_count in channel_counts:
            for precision in precisions:
                for batch_size in batch_sizes:
                    #predict_dataset = DataComponents.Predict_Dataset("Datasets/predict", 112, 24, 8, 1)
                    '''train_dataset = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters Anisotropic.csv",
                                                                64,
                                                                size[0], size[1], True, False, 0,
                                                                0,
                                                                1)
                    train_label_mean = train_dataset.get_label_mean()
                    train_contour_mean = torch.tensor(0.5)
                    unsupervised_train_dataset = DataComponents.UnsupervisedDataset("Datasets/unsupervised_train",
                                                                                    "Augmentation Parameters Anisotropic.csv",
                                                                                    64,
                                                                                    size[0], size[1])
                    train_dataset = DataComponents.CollectedDataset(train_dataset, unsupervised_train_dataset)
                    sampler = DataComponents.CollectedSampler(train_dataset, batch_size, unsupervised_train_dataset)
                    collate_fn = DataComponents.custom_collate
                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                               collate_fn=collate_fn, sampler=sampler,
                                                               num_workers=0)#, pin_memory=False, persistent_workers=True)
                    #meta_info = predict_dataset.__getmetainfo__()
                    #predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
                    val_dataset = DataComponents.ValDataset("Datasets/val", size[0], size[1], True, "Augmentation Parameters Anisotropic.csv")
                    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)

                    callbacks = []
                    model_checkpoint_last = pl.callbacks.ModelCheckpoint(dirpath="",
                                                                         filename="example_name",
                                                                         save_weights_only=True, enable_version_counter=False)
                    swa_callback = StochasticWeightAveraging(1e-5, 0.8, int(0.2 * 10 - 1))
                    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
                    callbacks.append(model_checkpoint_last)
                    callbacks.append(swa_callback)
                    arch_args = ('InstanceResidual_Recommended', channel_count, size[2], 3.33, True, train_label_mean, train_contour_mean)
                    model = PLModule(arch_args,
                                    True, False, 'Datasets/mid_visualiser/Ts-4c_visualiser.tif', True,
                                    False, False, False, True)
                    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, logger=TensorBoardLogger(f'lightning_logs', name=f'{size}-{channel_count}-{precision}-{batch_size}'),
                                         accelerator="gpu", enable_checkpointing=True, gradient_clip_val=0.2,
                                         precision=precision, enable_progress_bar=True, num_sanity_val_steps=0, callbacks=callbacks)
                                                                                                                      #FineTuneLearningRateFinder(min_lr=0.00001, max_lr=0.1, attr_name='initial_lr')])
                    # print(subprocess.run("tensorboard --logdir='lightning_logs'", shell=True))
                    #snap2 = tracemalloc.take_snapshot()
                    #top_stats = snap2.compare_to(snap1, 'lineno')
                    #for stat in top_stats[:20]:
                    #    print(stat)
                    #start_time = time.time()
                    trainer.fit(model,
                                val_dataloaders=val_loader,
                                train_dataloaders=train_loader)
                    torch.cuda.empty_cache()'''
                    model = PLModule.load_from_checkpoint("'results'/Kasthuri_connectomic_largefov.ckpt")
                    trainer = pl.Trainer(precision=precision, enable_progress_bar=True, logger=False, accelerator="gpu")
                    predict_dataset = DataComponents.Predict_Dataset('Datasets/predict',
                                                                     hw_size=160, depth_size=48,
                                                                     hw_overlap=16, depth_overlap=4)
                    predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
                    meta_info = predict_dataset.__getmetainfo__()
                    start_time = time.time()
                    predictions = trainer.predict(model, predict_loader)
                    end_time = time.time()
                    DataComponents.predictions_to_final_img_instance(predictions, meta_info, direc='Datasets/result',
                                                                     hw_size=160, depth_size=48,
                                                                     hw_overlap=16, depth_overlap=4, segmentation_mode='watershed')
    #model = PLModule.load_from_checkpoint('test.ckpt')
    '''predictions = trainer.predict(model, predict_loader)
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
    '''
    torch.save(model.state_dict(), 'placeholder.pth')
    
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    DataComponents.predictions_to_final_img_instance(predictions, meta_info, direc='Datasets/result',
                                                     hw_size=256, depth_size=64,
                                                     hw_overlap=32, depth_overlap=8)
'''
