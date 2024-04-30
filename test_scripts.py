from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from Components import DataComponents
import torch.utils.tensorboard
from pytorch_lightning.loggers import TensorBoardLogger
import run_semantic
from Networks import *
import subprocess
import threading
from lightning.pytorch.callbacks import EarlyStopping


def start_tensorboard():
    subprocess.run("tensorboard --logdir='/mnt/7018F20D48B6C548/PycharmProjects/Deeplearning/CV/lightning_logs'", shell=True)


torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"
tensorboard_thread = threading.Thread(target=start_tensorboard)
tensorboard_thread.daemon = True
tensorboard_thread.start()

def train_model(model, k):
    logger = TensorBoardLogger('lightning_logs', name=f'{model._get_name()}_fold_{k}')
    train_dataset = DataComponents.Cross_Validation_Dataset('Cross_Validate_Dataset', "Augmentation Parameters.csv",
                                                            k, True, 8)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dataset = DataComponents.Cross_Validation_Dataset('Cross_Validate_Dataset', "Augmentation Parameters.csv",
                                                          k, False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=0)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=40, verbose=False, mode="min")
    trainer = Trainer(max_epochs=1, log_every_n_steps=1, logger=logger,
                      accelerator="gpu", enable_checkpointing=False,
                      precision=32, callbacks=early_stop_callback, min_epochs=1)
    model = run_semantic.PLModuleSemantic(model, True, 8, 0.0001, True, False, None,
                                 False, False, False)
    trainer.fit(model,
                val_dataloaders=val_loader,
                train_dataloaders=train_loader)

# Aim: VRAM use <8192mb
model_list = [
                  (Iterations_New.Initial, 8),
                  (Iterations_New.Dropout_Middle, 21),
                  (Iterations_New.Dropout_All, 21),
                  (Iterations_New.Dropout_All_Low, 21),
                  (Iterations_New.InstanceNorm, 21),
                  (Iterations_New.Residual_Basic, 21),
                  (Iterations_New.Residual_Bottleneck, 38),
                  (Iterations_New.Residual_Bottleneck_Small_Neck, 34),
                  ##(Iterations_New.GroupedConv, 20),
                  ##(Iterations_New.GroupedConv_More, 20),
                  (Iterations_New.GhostModules, 28),
                  (Iterations_New.Squeeze_AND_Excite, 21),
                  (Iterations_New.Asymmetric, 21),
                  (Iterations_New.AsymmetricMore, 21),
                  (Iterations_New.SubpixelConvolutions, 23),
                  (Iterations_New.SimplifiedDecoder, 28),
                  (Iterations_New.HalfUNetDecoder, 49),
                  (Iterations_New.StridedConvDown, 21),
                  (Iterations_New.StridedConvDownKS3, 21),
                  (Iterations_New.PrePoolDouble, 21),
                  (Iterations_New.Chained2, 15),
                  (Iterations_New.NonResPreProcess, 16),
                  (Iterations_New.InitialWithELU, 21),
                  (Iterations_New.InitialWithPReLU, 21),
                  (Iterations_New.InitialWithGELU, 21)
              ]

#for model in model_list:
#    k = 0
    #while k < 4:
    #    repeat_list = [45, 45, 30, 30]
    #    i = 0
    #    while i < repeat_list[k]:
#    train_model(model[0](base_channels=model[1]), k)
    #        i = i + 1
    #    k = k + 1

k_to_test = [0,1,2,3,4]
for k in k_to_test:
    repeat_list = [1, 1, 1, 1, 1]
    i = 0
    while i < repeat_list[k]:
        train_model(model_list[0][0](base_channels=model_list[0][1]), k)
        i = i + 1
