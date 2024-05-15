import os
import time
import threading
import argparse
import subprocess
import webbrowser

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from run_semantic import PLModuleSemantic
from run_instance import PLModuleInstance
from Components import DataComponents
from Networks import *
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger


num_cpu_cores = os.cpu_count()
desired_num_workers = max(num_cpu_cores-1, 1)


def create_logger(args):
    logger = TensorBoardLogger(f'{args.tensorboard_path}', name='Run')
    return logger


def start_tensorboard(args):
    # Read the TENSORBOARD_PORT from the environment, or use the default
    tensorboard_port = os.environ.get('TENSORBOARD_PORT', 6006)
    subprocess.run(f"tensorboard --logdir={args.tensorboard_path} --port={str(tensorboard_port)}", shell=True)
    tensorboard_url = f'http://localhost:{tensorboard_port}'
    webbrowser.open(tensorboard_url)


def pick_arch(arch, base_channels, depth, z_to_xy_ratio, se):
    if arch == "HalfUNetBasic":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Basic', se)
    elif arch == "HalfUNetGhost":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Ghost', se)
    elif arch == "HalfUNetResidual":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'Residual', se)
    elif arch == "HalfUNetResidualBottleneck":
        return Semantic_HalfUNets.HalfUNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se)
    elif arch == "UNetBasic":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'Basic', se)
    elif arch == "UNetResidual":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'Residual', se)
    elif arch == "UNetResidualBottleneck":
        return Semantic_General.UNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se)
    elif arch == "SegNet":
        return Semantic_SegNets.Auto(base_channels, depth, z_to_xy_ratio, se)
    #elif arch == "Tiniest":
    #    return Testing_Models.Tiniest(base_channels, depth, z_to_xy_ratio)
    #elif arch == "SingleTopLayer":
    #    return Testing_Models.SingleTopLayer(base_channels, depth, z_to_xy_ratio, 'Basic', se)
    elif arch == "InstanceBasic":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'Basic', se)
    elif arch == "InstanceResidual":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'Residual', se)
    elif arch == "InstanceResidualBottleneck":
        return Instance_General.UNet(base_channels, depth, z_to_xy_ratio, 'ResidualBottleneck', se)


def start_work_flow(args):
    torch.set_float32_matmul_precision('medium')
    if 'Training' in args.workflow_box:
        if 'Semantic' in args.mode_box:
            train_dataset = DataComponents.TrainDataset(args.train_dataset_path, args.augmentation_csv_path, args.train_multiplier,
                                                        args.exclude_edge, args.exclude_edge_size_in, args.exclude_edge_size_out)
        else:
            train_dataset = DataComponents.TrainDatasetInstance(args.train_dataset_path, args.augmentation_csv_path,
                                                                args.train_multiplier, args.contour_map_width)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True,
                                  #num_workers=desired_num_workers, persistent_workers=True, pin_memory=True)
                                  num_workers=0, pin_memory=True)
        del train_dataset
        if 'Validation' in args.workflow_box:
            if 'Semantic' in args.mode_box:
                val_dataset = DataComponents.ValDataset(args.val_dataset_path, args.augmentation_csv_path)
            else:
                val_dataset = DataComponents.ValDatasetInstance(args.val_dataset_path, args.augmentation_csv_path, args.contour_map_width)
            val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                    #num_workers=desired_num_workers, persistent_workers=True, pin_memory=True)
                                    num_workers=0, pin_memory=True)

            del val_dataset
        else:
            val_loader = None
        if 'Test' in args.workflow_box:
            if 'Semantic' in args.mode_box:
                test_dataset = DataComponents.ValDataset(args.test_dataset_path, args.augmentation_csv_path)
            else:
                test_dataset = DataComponents.ValDatasetInstance(args.test_dataset_path, args.augmentation_csv_path, args.contour_map_width)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
            del test_dataset
    arch = pick_arch(args.model_architecture, args.model_channel_count, args.model_depth, args.model_z_to_xy_ratio, args.model_se)
    if ('Sparsely Labelled' in args.train_dataset_mode) or (args.exclude_edge):
        sparse_train = True
    else:
        sparse_train = False
    if 'Semantic' in args.mode_box:
        model = PLModuleSemantic(arch,
                                 #args.initial_lr, args.patience, args.min_lr,
                                 'Validation' in args.workflow_box, args.enable_mid_visualization,
                                 args.mid_visualization_input,
                                 sparse_train, 'Sparsely Labelled' in args.val_dataset_mode,
                                 'Sparsely Labelled' in args.test_dataset_mode, args.enable_tensorboard)
    else:
        model = PLModuleInstance(arch,
                                 #args.initial_lr, args.patience, args.min_lr,
                                 'Validation' in args.workflow_box, args.enable_mid_visualization,
                                 args.mid_visualization_input,
                                 sparse_train, 'Sparsely Labelled' in args.val_dataset_mode,
                                 'Sparsely Labelled' in args.test_dataset_mode, args.enable_tensorboard)
    del arch
    if args.read_existing_model:
        model.load_state_dict(torch.load(args.existing_model_path))
    if args.enable_tensorboard:
        tensorboard_thread = threading.Thread(target=start_tensorboard, args=[args])
        tensorboard_thread.daemon = True
        tensorboard_thread.start()
        logger = create_logger(args)
    else:
        logger = False
    if 'Training' in args.workflow_box:
        trainer = pl.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, logger=logger,
                             accelerator="gpu", enable_checkpointing=False,
                             precision=args.precision, enable_progress_bar=True, num_sanity_val_steps=0,
                             #profiler='simple',
                             )
        start_time = time.time()
        trainer.fit(model,
                    val_dataloaders=val_loader,
                    train_dataloaders=train_loader)
        del val_loader, train_loader
        end_time = time.time()
        print(f"Training Taken: {end_time - start_time} seconds")
    if 'Test' in args.workflow_box:
        trainer = pl.Trainer(precision=args.precision, enable_progress_bar=True, logger=logger, accelerator="gpu")
        trainer.test(model, dataloaders=test_loader)
        del test_loader
    if args.save_model:
        full = os.path.join(args.save_model_path, args.save_model_name)
        torch.save(model.state_dict(), full)
    if 'Predict' in args.workflow_box:
        trainer = pl.Trainer(precision=args.precision, enable_progress_bar=True, logger=False, accelerator="gpu")
        predict_dataset = DataComponents.Predict_Dataset(args.predict_dataset_path,
                                                         hw_size=args.predict_hw_size, depth_size=args.predict_depth_size,
                                                         hw_overlap=args.predict_hw_overlap, depth_overlap=args.predict_depth_overlap,
                                                         TTA_hw=args.TTA_xy)
        meta_info = predict_dataset.__getmetainfo__()
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
        del predict_dataset
        start_time = time.time()
        predictions = trainer.predict(model, predict_loader)
        end_time = time.time()
        del predict_loader
        print(f"Prediction Taken: {end_time - start_time} seconds")
        if 'Semantic' in args.mode_box:
            start_time = time.time()
            DataComponents.predictions_to_final_img(predictions, meta_info, direc=args.result_folder_path,
                                                    hw_size=args.predict_hw_size, depth_size=args.predict_depth_size,
                                                    hw_overlap=args.predict_hw_overlap, depth_overlap=args.predict_depth_overlap,
                                                    TTA_hw=args.TTA_xy)
            end_time = time.time()
            print(f"Converting and saving taken: {end_time - start_time} seconds")
        else:
            start_time = time.time()
            DataComponents.predictions_to_final_img_instance(predictions, meta_info, direc=args.result_folder_path,
                                                             hw_size=args.predict_hw_size, depth_size=args.predict_depth_size,
                                                             hw_overlap=args.predict_hw_overlap, depth_overlap=args.predict_depth_overlap,
                                                             TTA_hw=args.TTA_xy)
            end_time = time.time()
            print(f"Converting and saving taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Workflow")
    parser.add_argument("--workflow_box", nargs='+', choices=["Training", "Validation", "Test", "Predict"], default=[],
                        help="Workflows to enable")
    parser.add_argument("--mode_box", choices=["Semantic", "Instance"], default="Semantic",
                        help="Segmentation Mode")
    parser.add_argument("--train_dataset_path", type=str, default="Datasets/train", help="Train Dataset Path")
    parser.add_argument("--augmentation_csv_path", type=str, default="Augmentation Parameters.csv",
                        help="Csv File for Data Augmentation Settings")
    parser.add_argument("--train_multiplier", type=int, default=8, help="Train Multiplier (Repeats)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size")
    parser.add_argument("--max_epochs", type=int, default=40, help="Maximum Number of Epochs")
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable TensorBoard Logging")
    parser.add_argument("--tensorboard_path", type=str, default="lightning_logs",
                        help="Path to the folder which the log will be saved to")
    parser.add_argument("--val_dataset_path", type=str, default="Datasets/val", help="Validation Dataset Path")
    parser.add_argument("--test_dataset_path", type=str, default="Datasets/test", help="Test Dataset Path")
    parser.add_argument("--predict_dataset_path", type=str, default="Datasets/predict", help="Predict Dataset Path")
    parser.add_argument("--read_existing_model", action="store_true", help="Read Existing Model Weight File")
    parser.add_argument("--existing_model_path", type=str, default="", help="Path to Existing Model Weight File")
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16"], default="32", help="Precision")
    parser.add_argument("--save_model", action="store_true", help="Save Trained Model Weight")
    parser.add_argument("--save_model_name", type=str, default="example_name.pth",
                        help="File Name for Model Saved, including extension")
    parser.add_argument("--save_model_path", type=str, default=".", help="Path to Save the Model Weight")
    parser.add_argument("--predict_hw_size", type=int, default=128, help="Height and Width of each Patch (px)")
    parser.add_argument("--predict_depth_size", type=int, default=128, help="Depth of each Patch (px)")
    parser.add_argument("--predict_hw_overlap", type=int, default=8,
                        help="Expansion in Height and Width for each Patch (px)")
    parser.add_argument("--predict_depth_overlap", type=int, default=8, help="Expansion in Depth for each Patch (px)")
    parser.add_argument("--result_folder_path", type=str, default="Datasets/result", help="Result Folder Path")
    parser.add_argument("--enable_mid_visualization", action="store_true", help="Enable Visualization")
    parser.add_argument("--mid_visualization_input", type=str, default="Datasets/mid_visualiser/image.tif",
                        help="Path to the input image")
    parser.add_argument("--model_architecture", type=str,
                        help="Model Architecture")
    parser.add_argument("--model_channel_count", type=int, default=8, help="Base Channel Count")
    parser.add_argument("--model_depth", type=int, default=5, help="Model Depth")
    parser.add_argument("--model_z_to_xy_ratio", type=float, default=1.0, help="Enable Squeeze-and-Excitation plug-in")
    parser.add_argument("--model_se", action="store_true")
    parser.add_argument("--train_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"],
                        default="Fully Labelled", help="Dataset Mode")
    parser.add_argument("--exclude_edge", action="store_true", help="Mark pictures at object borders as unlabelled")
    parser.add_argument("--exclude_edge_size_in", type=int, default=1, help="Pixels to exclude (inward)")
    parser.add_argument("--exclude_edge_size_out", type=int, default=1, help="Pixels to exclude (outward)")
    parser.add_argument("--contour_map_width", type=int, default=1, help="Width of contour (outward)")
    parser.add_argument("--val_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"], default="Fully Labelled",
                        help="Dataset Mode")
    parser.add_argument("--test_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"],
                        default="Fully Labelled", help="Dataset Mode")
    parser.add_argument("--TTA_xy", action="store_true", help="Enable Test-Time Augmentation for xy dimension")

    args = parser.parse_args()
    start_work_flow(args)