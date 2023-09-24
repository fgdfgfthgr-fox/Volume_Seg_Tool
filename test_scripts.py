import lightning.pytorch as pl
import torch.utils.data
from Components import DataComponents
import torch.utils.tensorboard
from pytorch_lightning.loggers import TensorBoardLogger
import time
import Run
from Networks import *
import subprocess
import threading
from lightning.pytorch.callbacks import EarlyStopping


def start_tensorboard():
    subprocess.run("tensorboard --logdir='lightning_logs'", shell=True)


torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"
tensorboard_thread = threading.Thread(target=start_tensorboard)
tensorboard_thread.daemon = True
tensorboard_thread.start()
i = 0
time_list = []
while i < 4:
    k = 0
    while k < 2:
        logger = TensorBoardLogger('lightning_logs', name=f'It1_1_k={k}')
        train_dataset = DataComponents.Cross_Validation_Dataset('Cross_Validate_Dataset', "Augmentation Parameters.csv",
                                                                k, True, 10)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True,
                                                   num_workers=0)
        val_dataset = DataComponents.Cross_Validation_Dataset('Cross_Validate_Dataset', "Augmentation Parameters.csv",
                                                              k, False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1,
                                                 num_workers=0)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=25, verbose=False, mode="min")
        trainer = pl.Trainer(max_epochs=500, log_every_n_steps=1, logger=logger,
                             accelerator="gpu", enable_checkpointing=False,
                             precision=32, callbacks=early_stop_callback)
        model = Run.PLModule(iterations.Iteration_1_1(base_channels=8))
        start_time = time.time()
        trainer.fit(model,
                    val_dataloaders=val_loader,
                    train_dataloaders=train_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_list.append((elapsed_time))
        k = k + 1
    i = i + 1
print(f"It1_1 Average Training time = {sum(time_list) / len(time_list)}")
time_list.clear()