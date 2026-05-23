import math

import lightning.pytorch as pl
import torch
import torch._logging
import torch.utils.data
import torch.utils.tensorboard

from Components import DataComponents
from Components import SimMIM
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from Networks import *
from lightning.pytorch.callbacks import LearningRateMonitor
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
    def __init__(self, arch_args, enable_mid_visual, logging, mask_patch_size_multipliers):
        super().__init__()
        self.save_hyperparameters()
        self.patch_dim = arch_args[0]
        self.network = DiT.SwinTransformerForSimMIM(*arch_args)
        self.mim_loss_fn = DiT.SimMIM(self.network)
        self.enable_mid_visual = enable_mid_visual
        self.logging = logging
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []
        self.lr = 3e-2
        self.mask_patch_size_multipliers = mask_patch_size_multipliers

    def forward(self, image, mask):
        return self.mim_loss_fn(image, mask)

    def configure_optimizers(self):
        #fused = True if device == "cuda" else False
        param_groups = get_parameter_groups_with_muon(self, weight_decay=0.001)
        optimizer = AdaMuon(param_groups, lr=self.lr, weight_decay=0.001, adamw_lr=3e-4, adamw_wd=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=30,
                                                               threshold_mode='rel',
                                                               cooldown=0, min_lr=[1e-3, 1e-5, 1e-3, 1e-5])
        metrics = "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": metrics, "interval": "epoch", "frequency": 1},
        }

    @staticmethod
    def to_visualisation_img(tensor):
        tensor = tensor[0:1, :, 0:1, :, :].squeeze([0, 1])
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)

    def _step(self, batch):
        img = batch[0]
        mask = SimMIM.generate_simmim_mask(img, (s * self.mask_patch_size_multipliers for s in self.patch_dim), self.patch_dim, (0.5, 0.75))
        loss, x_rec = self.forward(img, mask)
        with torch.no_grad():
            if self.global_step % 64 == 0 and self.enable_mid_visual:
                original_shape = img.shape
                unfolded_img = DiT.unfold(img, self.patch_dim)
                mask2 = mask.unsqueeze(-1)
                unfolded_img = (unfolded_img * (1-mask2))
                folded_img = DiT.fold(unfolded_img, self.patch_dim, original_shape)
                mid_visual_img = self.to_visualisation_img(folded_img)
                mid_visual_gt = self.to_visualisation_img(img)
                self.logger.experiment.add_image(f'Visualization/Input', mid_visual_img, self.global_step)
                self.logger.experiment.add_image(f'Visualization/Ground Truth', mid_visual_gt, self.global_step)
                mid_visual_out = self.to_visualisation_img(x_rec)
                self.logger.experiment.add_image(f'Visualization/Output', mid_visual_out, self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, logger=False)
        self.train_metrics.append((loss,))
        return {'loss': loss}



    def log_metrics(self, prefix, metrics_list):
        if metrics_list:
            epoch_averages = torch.stack([torch.tensor(metrics) for metrics in metrics_list]).nanmean(dim=0)
            self.logger.experiment.add_scalar(f"{prefix}/Loss", epoch_averages[0], self.current_epoch)

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
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["train_loss"])

    #def on_before_optimizer_step(self, optimizer):
    #    norms = grad_norm(self.network, norm_type=2)
    #    self.log_dict(norms, logger=True)


if __name__ == "__main__":
    #tracemalloc.start()
    #snap1 = tracemalloc.take_snapshot()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    unsupervised_train_dataset = DataComponents.UnsupervisedDataset("Datasets/unsupervised_train",
                                                                    "Augmentation Parameters Anisotropic.csv",
                                                                    128,
                                                                    180, 56)
    train_loader = torch.utils.data.DataLoader(dataset=unsupervised_train_dataset, batch_size=2,
                                               num_workers=16, pin_memory=True, persistent_workers=True)
    callbacks = []
    #swa_callback = StochasticWeightAveraging([1e-3, 1e-5, 1e-3, 1e-5], 0.8, int(0.2 * 10 - 1))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    #callbacks.append(swa_callback)
    arch_args = ((2,5,5), 4, 9, False)
    model = PLModule(arch_args, True, True, 4)
    trainer = pl.Trainer(max_epochs=80, log_every_n_steps=1, logger=TensorBoardLogger(f'lightning_logs', name=f'SimMIM_Pretraining_4_lowermaskratio'),
                         accelerator="gpu", enable_checkpointing=True,# gradient_clip_val=0.2,
                         precision="bf16-mixed", enable_progress_bar=True, num_sanity_val_steps=0, callbacks=callbacks)
    trainer.fit(model,
                train_dataloaders=train_loader)
    torch.save(model.network.state_dict(), f"trained_model/simmim_tester.pt")