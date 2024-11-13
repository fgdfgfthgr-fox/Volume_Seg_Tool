from lightning.pytorch import Trainer
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from Components import DataComponents
import torch.utils.tensorboard
from pytorch_lightning.loggers import TensorBoardLogger
from pl_module import PLModule
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor
import subprocess
import threading


def start_tensorboard():
    subprocess.run("tensorboard --logdir='lightning_logs/test'", shell=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.version.cuda:
        print('Optimising computing and memory use via cuDNN! (NVIDIA GPU only).')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    #tensorboard_thread = threading.Thread(target=start_tensorboard)
    #tensorboard_thread.daemon = True
    #tensorboard_thread.start()
    val_dataset = DataComponents.ValDataset("Datasets/val", 128, 64, False, "Augmentation Parameters.csv", 2)
    train_dataset_pos = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters.csv",
                                                    64,
                                                    128, 64, False, False, 0,
                                                    0,
                                                    2, 'positive')
    train_dataset_neg = DataComponents.TrainDataset("Datasets/train", "Augmentation Parameters.csv",
                                                    64,
                                                    128, 64, False, False, 0,
                                                    0,
                                                    2, 'negative')
    loader_for_lr = torch.utils.data.DataLoader(dataset=train_dataset_pos, batch_size=2, num_workers=16, pin_memory=True, persistent_workers=True)
    train_label_mean = train_dataset_pos.get_label_mean()
    #train_contour_mean = train_dataset_pos.get_contour_mean()
    unsupervised_train_dataset = DataComponents.UnsupervisedDataset("Datasets/unsupervised_train",
                                                                    "Augmentation Parameters.csv",
                                                                    64,
                                                                    128, 64)
    train_dataset = DataComponents.CollectedDataset(train_dataset_pos, train_dataset_neg, unsupervised_train_dataset)
    sampler = DataComponents.CollectedSampler(train_dataset, 2, unsupervised_train_dataset)
    collate_fn = DataComponents.custom_collate
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2,
                                               collate_fn=collate_fn, sampler=sampler,
                                               num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=2, num_workers=0, pin_memory=True)
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="", filename="test", mode="max",
                                                    monitor="Val_epoch_dice", save_weights_only=True,
                                                    enable_version_counter=False)
    arch_args = ('InstanceResidual', 8, 4, 1, True, train_label_mean, 0.5)

    def train_model():
        model = PLModule(arch_args,
            True, True, 'Datasets/mid_visualiser/Ts-4c_visualiser.tif', False,
            False, False, False, True)
        trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1, logger=TensorBoardLogger(f'lightning_logs', name='test'),
                             accelerator="gpu", enable_checkpointing=True,
                             precision='bf16-mixed', enable_progress_bar=True, num_sanity_val_steps=0, callbacks=[model_checkpoint, LearningRateMonitor(logging_interval='epoch')])
        #tuner = Tuner(trainer)
        #lr_finder = tuner.lr_find(model, loader_for_lr, min_lr=1e-5, max_lr=0.002)
        #new_lr = lr_finder.suggestion()
        #print(f'Learning Rate set to: {new_lr}.')
        #model.hparams.lr = new_lr
        trainer.fit(model,
                    val_dataloaders=val_loader,
                    train_dataloaders=train_loader)
    #i = 0.1
    #while i <= 1.5:
    train_model()
    #    i += 0.1