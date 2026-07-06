import tifffile
import math

import lightning.pytorch as pl
import torch
import torch.utils.data
import torch.utils.tensorboard

import Components.Datasets
from Components import Metrics
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from Networks import *
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.utilities import grad_norm
from pytorch_optimizer.optimizer import AdaMuon


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parameter_groups_with_muon(model, weight_decay=0.0001):
    no_decay_keywords = ["bias", "bn", "batch_norm", "layer_norm", "norm", "RMSNorm"]

    # Create the 4 groups
    decay_muon_params = []  # decay + muon
    decay_no_muon_params = []  # decay + no muon
    no_decay_muon_params = []  # no decay + muon
    no_decay_no_muon_params = []  # no decay + no muon

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine decay status
        requires_decay = not any(no_decay_keyword in name for no_decay_keyword in no_decay_keywords)

        # Determine muon status (hidden weights with ndim == 2)
        is_muon = param.ndim == 2

        # Assign to appropriate group
        if requires_decay and is_muon:
            decay_muon_params.append(param)
        elif requires_decay and not is_muon:
            decay_no_muon_params.append(param)
        elif not requires_decay and is_muon:
            no_decay_muon_params.append(param)
        else:  # not requires_decay and not is_muon
            no_decay_no_muon_params.append(param)

    return [
        # Group 1: decay + muon
        {'params': decay_muon_params, 'weight_decay': weight_decay, 'use_muon': True},

        # Group 2: decay + no muon
        {'params': decay_no_muon_params, 'betas': (0.9, 0.95), 'weight_decay': weight_decay, 'use_muon': False},

        # Group 3: no decay + muon
        {'params': no_decay_muon_params, 'weight_decay': 0.0, 'use_muon': True},

        # Group 4: no decay + no muon
        {'params': no_decay_no_muon_params, 'betas': (0.9, 0.95), 'weight_decay': 0.0, 'use_muon': False}
    ]

class PLModule(pl.LightningModule):
    def __init__(self, arch_args, enable_val, enable_mid_visual, instance_mode,
                 use_sparse_label_train, use_sparse_label_val, use_sparse_label_test, logging):
        super().__init__()
        self.save_hyperparameters()
        self.network = DiT.SwinTransformer(*arch_args)
        self.enable_val = enable_val
        self.enable_mid_visual = enable_mid_visual
        self.instance_mode = instance_mode
        self.use_sparse_label_train = use_sparse_label_train
        self.use_sparse_label_val = use_sparse_label_val
        self.use_sparse_label_test = use_sparse_label_test
        self.logging = logging
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []
        self.lr = 1e-2
        self.pixel_ramp_steps = 2048
        self.unsupervised_weight = 0.1
        self.p_loss_fn = Metrics.BinaryMetrics("focal")
        self.c_loss_fn = Metrics.BinaryMetrics("dice+bce")

        self.dice_threshold_reached = False
        self.starting_step = None
        self.require_next_mid_visual = False

    def forward(self, image):
        return self.network(image)

    def compute_ramp_up_weight(self):
        # Check if the dice score threshold has been reached
        if not self.dice_threshold_reached:
            return 0.0  # No ramp-up until dice score exceeds threshold

        # Get the current global step and compute the ramp-up weight
        if self.starting_step is None:
            self.starting_step = self.global_step
        current_step = self.global_step - self.starting_step
        if current_step < self.pixel_ramp_steps:
            return math.e ** (-5 * ((1 - (current_step / self.pixel_ramp_steps)) ** 2)) * self.unsupervised_weight
        else:
            return self.unsupervised_weight

    def configure_optimizers(self):
        #fused = True if device == "cuda" else False
        param_groups = get_parameter_groups_with_muon(self, weight_decay=0.001)
        optimizer = AdaMuon(param_groups, lr=self.lr, weight_decay=0.001, adamw_lr=3e-4, adamw_wd=0.001, foreach=True)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=30,
                                                               threshold_mode='rel',
                                                               cooldown=0, min_lr=[1e-3, 1e-5, 1e-3, 1e-5])'''
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, (50,), 0.5)
        metrics = "val_loss" if self.enable_val else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": metrics, "interval": "epoch", "frequency": 1},
        }

    @staticmethod
    def to_visualisation_img(tensor, norm='None'):
        tensor = tensor[0:1, :, 0:1, :, :].squeeze([0, 1])
        if norm == 'None':
            return tensor
        elif norm == 'Sigmoid':
            return torch.sigmoid(tensor)
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)

    @staticmethod
    def entropy_preprocess(value):
        return 0.999 * torch.sigmoid(value) + 5e-4

    @torch.no_grad()
    @torch.compiler.disable()
    def visualise_instance(self, img, lab, contour, p_out, c_out):
        if self.training and ((self.global_step % 128 == 0 and self.enable_mid_visual) or self.require_next_mid_visual):
            mid_img = self.to_visualisation_img(img, 'Norm')
            mid_lab = self.to_visualisation_img(lab, 'None')
            mid_contour = self.to_visualisation_img(contour, 'None')
            mid_p_output = self.to_visualisation_img(p_out, 'Sigmoid')
            mid_c_output = self.to_visualisation_img(c_out, 'Sigmoid')
            self.logger.experiment.add_image(f'Visualization/Input', mid_img, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Pixel Ground Truth', mid_lab, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Contour Ground Truth', mid_contour, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Model Output (Pixel)', mid_p_output, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Model Output (Contour)', mid_c_output, self.global_step)

    @torch.no_grad()
    @torch.compiler.disable()
    def visualise_semantic(self, img, lab, out):
        if self.training and self.global_step % 128 == 0 and self.enable_mid_visual:
            mid_img = self.to_visualisation_img(img, 'Norm')
            mid_lab = self.to_visualisation_img(lab, 'None')
            mid_output = self.to_visualisation_img(out, 'Sigmoid')
            self.logger.experiment.add_image(f'Visualization/Input', mid_img, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Ground Truth', mid_lab, self.global_step)
            self.logger.experiment.add_image(f'Visualization/Model Output', mid_output, self.global_step)

    def unsupervised_step(self, batch):
        pixel_ramp_weight = self.compute_ramp_up_weight()
        img = batch[0]
        if self.global_step % 128 == 0:
            self.require_next_mid_visual = True
        if self.instance_mode:
            p_output, c_output = self.forward(img)
            sigmoid_p_output = self.entropy_preprocess(p_output)
            # Reducing the entropy of contour seems to perform worse
            entropy = (-sigmoid_p_output * torch.log(sigmoid_p_output)).mean()
            entropy_loss = entropy * pixel_ramp_weight
            return entropy_loss, *(torch.nan,) * 12, entropy
        else:
            output = self.forward(img)
            sigmoid_outputs = self.entropy_preprocess(output)
            entropy = (-sigmoid_outputs * torch.log(sigmoid_outputs)).mean()
            entropy_loss = entropy * pixel_ramp_weight
            return entropy_loss, *(torch.nan,) * 6, entropy

    def _step(self, batch, sparse):
        if self.instance_mode:
            img, lab, contour = batch
            p_output, c_output = self.forward(img)
            p_loss, p_i, p_u, p_tp, p_fn, p_tn, p_fp = self.p_loss_fn(p_output, lab, False)
            c_loss, c_i, c_u, c_tp, c_fn, c_tn, c_fp = self.c_loss_fn(c_output, contour, False)
            loss = p_loss + c_loss
            self.visualise_instance(img, lab, contour, p_output, c_output)
            return loss, p_i, c_i, p_u, c_u, p_tp, c_tp, p_fn, c_fn, p_tn, c_tn, p_fp, c_fp, torch.nan
        else:
            img, lab = batch
            output = self.forward(img)
            loss, i, u, tp, fn, tn, fp = self.p_loss_fn(output, lab, sparse)
            self.visualise_semantic(img, lab, output)
            return loss, i, u, tp, fn, tn, fp, torch.nan

    def training_step(self, batch, batch_idx):
        if self.instance_mode:
            if len(batch) == 3:
                result_tuple = self._step(batch, False)
            elif self.dice_threshold_reached:
                result_tuple = self.unsupervised_step(batch)
            else:
                return None
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        else:
            if len(batch) == 2:
                result_tuple = self._step(batch, self.use_sparse_label_train)
            elif self.dice_threshold_reached:
                result_tuple = self.unsupervised_step(batch)
            else:
                return None
            self.log("train_loss", result_tuple[0], logger=False)
            self.train_metrics.append(result_tuple)
        return {'loss': result_tuple[0]}

    def validation_step(self, batch, batch_idx):
        if self.instance_mode:
            result_tuple = self._step(batch, False)
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

    @staticmethod
    def apply_tta_augmentation(data, i):
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

    @staticmethod
    def remove_padding(tensor, hw_overlap, depth_overlap):
        if hw_overlap or depth_overlap:
            tensor = tensor[:, :, (depth_overlap if depth_overlap else slice(None)):
                                   -(depth_overlap if depth_overlap else slice(None)),
                                   (hw_overlap if hw_overlap else slice(None)):
                                   -(hw_overlap if hw_overlap else slice(None)),
                                   (hw_overlap if hw_overlap else slice(None)):
                                   -(hw_overlap if hw_overlap else slice(None))]
        return tensor

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch, hw_overlap, depth_overlap = batch
        TTA_results = []
        for i in range(8):
            aug_batch = self.apply_tta_augmentation(batch, i)
            outputs = self.forward(aug_batch)

            if isinstance(outputs, tuple):
                p_outputs = torch.sigmoid(outputs[0]).to(torch.float16)
                #p_outputs = torch.sigmoid(torch.mean(torch.stack(outputs[0], dim=0), dim=0)).to(torch.float16)
                c_outputs = torch.sigmoid(outputs[1]).to(torch.float16)
                p_outputs = self.remove_padding(self.apply_tta_augmentation(p_outputs, i), hw_overlap, depth_overlap)
                c_outputs = self.remove_padding(self.apply_tta_augmentation(c_outputs, i), hw_overlap, depth_overlap)
                TTA_results.append((p_outputs, c_outputs))
            else:
                p_outputs = torch.sigmoid(outputs).to(torch.float16)
                #p_outputs = torch.sigmoid(torch.mean(torch.stack(outputs, dim=0), dim=0)).to(torch.float16)
                p_outputs = self.remove_padding(self.apply_tta_augmentation(p_outputs, i), hw_overlap, depth_overlap)
                TTA_results.append(p_outputs)

        if isinstance(TTA_results[0], tuple):
            p, c = zip(*TTA_results)
            #return torch.mean(torch.stack(p, dim=0), dim=0), torch.mean(torch.stack(c, dim=0), dim=0)
            p = torch.mean(torch.stack(p, dim=0), dim=0)
            c = torch.mean(torch.stack(c, dim=0), dim=0)
            tifffile.imwrite(f'Datasets/prediction_cache/Pixels_{batch_idx}.tiff', data=p.cpu().numpy(), compression='zlib')
            tifffile.imwrite(f'Datasets/prediction_cache/Contour_{batch_idx}.tiff', data=c.cpu().numpy(), compression='zlib')
        else:
            #return torch.mean(torch.stack(TTA_results, dim=0), dim=0)
            out = torch.mean(torch.stack(TTA_results, dim=0), dim=0)
            tifffile.imwrite(f'Datasets/prediction_cache/Semantic_{batch_idx}.tiff', data=out.cpu().numpy())

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
                if epoch_averages[13] != 0 and not torch.any(torch.isnan(epoch_averages[13])):
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
                self.logger.experiment.add_scalar(f"{prefix}/Sensitivity", sensitivity, self.current_epoch)
                self.logger.experiment.add_scalar(f"{prefix}/Specificity", specificity, self.current_epoch)
                if epoch_averages[7] != 0 and not torch.any(torch.isnan(epoch_averages[7])):
                    self.logger.experiment.add_scalar(f"{prefix}/Entropy", epoch_averages[7], self.current_epoch)
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
    torch.backends.cudnn.enabled = False
    sizes = [(144, 144)]
    precisions = ['bf16-mixed']
    batch_sizes = [2]
    for size in sizes:
        for precision in precisions:
            for batch_size in batch_sizes:
                #predict_dataset = DataComponents.Predict_Dataset("Datasets/predict", 112, 24, 8, 1)
                '''train_dataset = Components.Datasets.TrainDataset("Datasets/train",
                                                            "Augmentation Parameters Anisotropic.csv",
                                                                 32,
                                                                 size[0], size[1], False, False, 'default')
                unsupervised_train_dataset = Components.Datasets.UnsupervisedDataset("Datasets/unsupervised_train",
                                                                                "Augmentation Parameters Anisotropic.csv",
                                                                                     64,
                                                                                     size[0], size[1])
                train_dataset = DataComponents.CollectedDataset(train_dataset, None)
                sampler = DataComponents.CollectedSampler(train_dataset, batch_size, None)
                collate_fn = DataComponents.custom_collate
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                           collate_fn=collate_fn, sampler=sampler,
                                                           num_workers=8, pin_memory=True, persistent_workers=True)
                # meta_info = predict_dataset.__getmetainfo__()
                # predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
                #val_dataset = DataComponents.ValDataset("Datasets/val", size[0], size[1], True,
                #                                        "Augmentation Parameters Anisotropic.csv")
                #val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)

                callbacks = []
                model_checkpoint_last = pl.callbacks.ModelCheckpoint(dirpath="trained_model",
                                                                     filename="example_name",
                                                                     save_weights_only=True, enable_version_counter=False)
                #swa_callback = StochasticWeightAveraging([1e-3, 1e-5, 1e-3, 1e-5], 0.8, int(0.2 * 10 - 1))
                callbacks.append(LearningRateMonitor(logging_interval='epoch'))
                callbacks.append(model_checkpoint_last)
                #callbacks.append(swa_callback)
                arch_args = ((4,4,4), 4, 10, False)
                model = PLModule(arch_args,
                                False, True, False,
                                False, False, False, True)
                trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1, logger=TensorBoardLogger(f'lightning_logs', name=f'dit-4,8ps-384d-4layers-2heads-pcseperate-swin-64-window4'),
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
                            #val_dataloaders=val_loader,
                            train_dataloaders=train_loader)
                torch.cuda.empty_cache()'''