import os
import math
import gc
import time
import threading
import argparse
import subprocess
import webbrowser
import torch
import multiprocessing

import lightning.pytorch as pl

from torch.utils.data import DataLoader

from pl_module_dit import PLModule
from Components import Utils, Datasets, Datasets_Chunked
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_logger(args):
    logger = TensorBoardLogger(f'{args.tensorboard_path}', name='Run')
    return logger


def start_tensorboard(args):
    # Read the TENSORBOARD_PORT from the environment, or use the default
    tensorboard_port = os.environ.get('TENSORBOARD_PORT', 6006)
    subprocess.run(f"tensorboard --logdir={args.tensorboard_path} --port={str(tensorboard_port)}", shell=True)
    tensorboard_url = f'http://localhost:{tensorboard_port}'
    webbrowser.open(tensorboard_url)

def generalised_train(TD, UTD, VD, TeD, instance_mode, desired_num_workers, persistent_workers):
    assert args.train_multiplier != "None", "Received None for train_multiplier! Did you forgot to press the 'Calculate Training Repeats and Epochs' button?'"
    train_dataset = TD(args.train_dataset_path, args.augmentation_csv_path,
                       int(args.train_multiplier),
                       args.hw_size, args.d_size, instance_mode,
                       args.contour_map_width, args.train_key_name)
    if args.enable_unsupervised:
        unsupervised_train_dataset = UTD(args.unsupervised_train_dataset_path,
                                         args.augmentation_csv_path, int(args.unsupervised_train_multiplier),
                                         args.hw_size, args.d_size, args.train_key_name)
    else:
        unsupervised_train_dataset = None
    train_dataset = Utils.CollectedDataset(train_dataset, unsupervised_train_dataset)
    sampler = Utils.CollectedSampler(train_dataset, args.batch_size, unsupervised_train_dataset)
    collate_fn = Utils.custom_collate
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              collate_fn=collate_fn, sampler=sampler,
                              num_workers=desired_num_workers, persistent_workers=persistent_workers)
                              #num_workers=0, persistent_workers=False)
    del train_dataset
    if 'Validation' in args.workflow_box:
        val_dataset = VD(args.val_dataset_path, args.hw_size, args.d_size, instance_mode,
                         args.contour_map_width, args.val_key_name)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                                #num_workers=12, persistent_workers=True, pin_memory=True)
                                num_workers=0, pin_memory=True)
        del val_dataset
    else:
        val_loader = None
    if 'Test' in args.workflow_box:
        test_dataset = TeD(args.test_dataset_path, args.hw_size, args.d_size, instance_mode,
                           args.contour_map_width, args.test_key_name)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
        del test_dataset
    gc.collect()
    arch_args = ((args.model_patch_size_z, args.model_patch_size_xy, args.model_patch_size_xy), 4,
                 args.model_depth * args.model_depth_multiplier, instance_mode)
    if 'Sparsely Labelled' in args.train_dataset_mode:
        sparse_train = True
    else:
        sparse_train = False
    if args.read_existing_model:
        model = PLModule.load_from_checkpoint(args.existing_model_path, weights_only=False)
    else:
        model = PLModule(arch_args,
                         'Validation' in args.workflow_box, args.mid_visualization, instance_mode,
                         sparse_train, 'Sparsely Labelled' in args.val_dataset_mode,
                         'Sparsely Labelled' in args.test_dataset_mode, args.enable_tensorboard)
    if 'Training' in args.workflow_box:
        logger = create_logger(args)
    else:
        logger = False
    if 'Training' in args.workflow_box:
        '''if 'Validation' in args.workflow_box:
            to_monitor = 'Val_epoch_dice'
        else:
            to_monitor = 'Train_epoch_dice'''
        callbacks = []
        '''model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f"{args.save_model_path}",
                                                        filename=f"{args.save_model_name}",
                                                        mode="max", monitor=to_monitor,
                                                        save_weights_only=True, enable_version_counter=False)'''
        model_checkpoint_last = pl.callbacks.ModelCheckpoint(dirpath=f"{args.save_model_path}",
                                                             filename=f"{args.save_model_name}",
                                                             save_weights_only=False, enable_version_counter=False)
        swa_callback = StochasticWeightAveraging([1e-3, 1e-5, 1e-3, 1e-5], 0.8, int(0.2*int(args.num_epochs)-1))
        print(f'SWA starts at {int(0.8*int(args.num_epochs))}\n')
        if logger:
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        callbacks.append(model_checkpoint_last)
        callbacks.append(swa_callback)
        trainer = pl.Trainer(max_epochs=int(args.num_epochs), log_every_n_steps=1, logger=logger,
                             accelerator="gpu", enable_checkpointing=True,
                             precision=args.precision, enable_progress_bar=True, num_sanity_val_steps=0,
                             gradient_clip_val=5,
                             callbacks=callbacks,
                             #profiler='simple',
                             )
        start_time = time.time()
        if args.read_existing_model:
            trainer.fit(model,
                        ckpt_path=args.existing_model_path,
                        val_dataloaders=val_loader,
                        train_dataloaders=train_loader)
        else:
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

def load_data_and_predict(PD):
    if 'Training' in args.workflow_box:
        path = f"{args.save_model_path}/{args.save_model_name}.ckpt"
    elif args.read_existing_model:
        path = args.existing_model_path
    else:
        assert "No model specified for prediction!"
    model = PLModule.load_from_checkpoint(path, weights_only=False)
    trainer = pl.Trainer(precision=args.precision, enable_progress_bar=True, logger=False, accelerator="gpu")
    predict_dataset = PD(args.predict_dataset_path,
                         hw_size=args.predict_hw_size, depth_size=args.predict_depth_size,
                         hw_overlap=args.predict_hw_overlap, depth_overlap=args.predict_depth_overlap,
                         hdf5_key=args.predict_key_name)
    meta_info = predict_dataset.__getmetainfo__()
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
    start_time = time.time()
    trainer.predict(model, predict_loader, return_predictions=False)
    end_time = time.time()
    print(f"Prediction Taken: {end_time - start_time} seconds")
    return meta_info

def stitch(indexed_files, image_meta, current_offset, instance_mode):
    out_file_name = "Mask_" + image_meta[0]
    depth, height, width = image_meta[1][1:]
    depth_multiplier = math.ceil(depth / args.predict_depth_size)
    height_multiplier = math.ceil(height / args.predict_hw_size)
    width_multiplier = math.ceil(width / args.predict_hw_size)
    total_multiplier = depth_multiplier * height_multiplier * width_multiplier
    dim_info = depth, height, width, depth_multiplier, height_multiplier, width_multiplier, total_multiplier
    if instance_mode:
        stitched_volume_p = Utils.stitch_output_volume(indexed_files[0], dim_info, args.predict_hw_size,
                                                       args.predict_depth_size, current_offset)
        stitched_volume_c = Utils.stitch_output_volume(indexed_files[1], dim_info, args.predict_hw_size,
                                                       args.predict_depth_size, current_offset)
        current_offset += total_multiplier
        return ((stitched_volume_p, out_file_name), (stitched_volume_c, out_file_name)), current_offset
    else:
        stitched_volume = Utils.stitch_output_volume(indexed_files, dim_info, args.predict_hw_size,
                                                     args.predict_depth_size, current_offset)
        current_offset += total_multiplier
        return (stitched_volume, out_file_name), current_offset


def save_stitched_volume(stitched_volume, instance_mode):
    if args.instance_seg_mode:
        mode = 'watershed'
    else:
        mode = 'simple'
    if instance_mode:
        Utils.predictions_to_final_img_instance(stitched_volume[0], stitched_volume[1], direc=args.result_folder_path,
                                                segmentation_mode=mode, dynamic=args.watershed_dynamic,
                                                pixel_reclaim=args.pixel_reclaim)
    else:
        Utils.predictions_to_final_img(stitched_volume, direc=args.result_folder_path)

def delete_indexed_files(indexed_files):
    for _, file_path in indexed_files:
        try:
            file_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete {file_path}: {e}")


def start_work_flow(args):
    torch.set_float32_matmul_precision('medium')
    if torch.cuda.is_available() and torch.version.cuda:
        print('Optimising computing and memory use via cuDNN! (NVIDIA GPU only).')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    elif torch.cuda.is_available() and torch.version.hip:
        print('Optimising computing using TunableOp! (AMD GPU only).')
        torch.backends.cudnn.enabled = False
        #os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
        #torch.cuda.tunable.enable()
        #torch.cuda.tunable.set_filename('TunableOp_results')
    if args.train_offload:
        TD = Datasets_Chunked.TrainDatasetChunked
        UTD = Datasets_Chunked.UnsupervisedDatasetChunked
    else:
        TD = Datasets.TrainDataset
        UTD = Datasets.UnsupervisedDataset
    if args.val_offload:
        VD = Datasets_Chunked.ValDatasetChunked
    else:
        VD = Datasets.ValDataset
    if args.test_offload:
        TeD = Datasets_Chunked.ValDatasetChunked
    else:
        TeD = Datasets.ValDataset
    if args.predict_offload:
        PD = Datasets_Chunked.PredictDatasetChunked
    else:
        PD = Datasets.PredictDataset

    if 'Semantic' in args.segmentation_mode:
        instance_mode = False
    else:
        instance_mode = True
    if not args.memory_saving_mode:
        desired_num_workers = min(max((os.cpu_count()//2)-1, 1), 32)
        persistent_workers = True
    else:
        desired_num_workers = 0
        persistent_workers = False

    if 'Training' in args.workflow_box:
        tensorboard_thread = threading.Thread(target=start_tensorboard, args=[args])
        tensorboard_thread.daemon = True
        tensorboard_thread.start()
        generalised_train(TD, UTD, VD, TeD, instance_mode, desired_num_workers, persistent_workers)

    if 'Predict' in args.workflow_box:
        start_time = time.time()
        meta_info = load_data_and_predict(PD)
        if instance_mode:
            indexed_files_p = Utils.load_indexed_files_for_stitching("Pixels_")
            indexed_files_c = Utils.load_indexed_files_for_stitching("Contour_")
            indexed_files = (indexed_files_p, indexed_files_c)
        else:
            indexed_files = Utils.load_indexed_files_for_stitching("Semantic_")
        current_offset = 0
        for image_meta in meta_info:
            stitched_volume, current_offset = stitch(indexed_files, image_meta, current_offset, instance_mode)
            save_stitched_volume(stitched_volume, instance_mode)
        end_time = time.time()
        print(f"Converting and saving taken: {end_time - start_time} seconds")
        if instance_mode:
            delete_indexed_files(indexed_files_p)
            delete_indexed_files(indexed_files_c)
        else:
            delete_indexed_files(indexed_files)



if __name__ == "__main__":
    # To prevent some import errors that might showup
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Deep Learning Workflow")
    # The default values are for the coverage test using the default data
    parser.add_argument("--workflow_box", nargs='+', choices=["Training", "Validation", "Test", "Predict"], default=["Training", "Validation", "Predict"],
                        help="Workflows to enable")
    parser.add_argument("--segmentation_mode", choices=["Semantic", "Instance"], default="Semantic",
                        help="Segmentation Mode")
    parser.add_argument("--train_dataset_path", type=str, default="Datasets/train", help="Train Dataset Path")
    parser.add_argument("--augmentation_csv_path", type=str, default="Augmentation Parameters Anisotropic.csv",
                        help="Csv File for Data Augmentation Settings")
    parser.add_argument("--train_multiplier", type=str, default="128", help="Train Multiplier (Repeats)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--pairing_samples", default=True, action="store_true", help="Pairing positive and negative samples in a batch")
    parser.add_argument("--num_epochs", type=str, default="5", help="Number of Epochs") # Five epoch is enough
    parser.add_argument("--enable_tensorboard", default=True, action="store_true", help="Enable TensorBoard Logging")
    parser.add_argument("--enable_unsupervised", action="store_true", help="Enable Unsupervised Pretraining")
    parser.add_argument("--memory_saving_mode", action="store_true", help="Try save some system memory by dataloading on just single core")
    parser.add_argument("--unsupervised_train_dataset_path", type=str, default="Datasets/unsupervised_train", help="Unsupervised Dataset Path")
    parser.add_argument("--unsupervised_train_multiplier", type=str, default="128", help="Unsupervised Train Multiplier (Repeats)")
    parser.add_argument("--tensorboard_path", type=str, default="lightning_logs",
                        help="Path to the folder which the log will be saved to")
    parser.add_argument("--val_dataset_path", type=str, default="Datasets/val", help="Validation Dataset Path")
    parser.add_argument("--test_dataset_path", type=str, default="Datasets/test", help="Test Dataset Path")
    parser.add_argument("--predict_dataset_path", type=str, default="Datasets/predict", help="Predict Dataset Path")
    parser.add_argument("--read_existing_model", action="store_true", help="Read Existing Model Weight File")
    parser.add_argument("--existing_model_path", type=str, default="", help="Path to Existing Model Weight File")
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16-mixed"], default="32", help="Precision")
    parser.add_argument("--save_model_name", type=str, default="example_name",
                        help="File Name for Model Saved, does not include extension")
    parser.add_argument("--save_model_path", type=str, default="trained_model", help="Path to Save the Model Weight")
    parser.add_argument("--train_key_name", type=str, default=".", help="hdf5 dataset name")
    parser.add_argument("--val_key_name", type=str, default=".", help="hdf5 dataset name")
    parser.add_argument("--test_key_name", type=str, default=".", help="hdf5 dataset name")
    parser.add_argument("--predict_key_name", type=str, default=".", help="hdf5 dataset name")
    parser.add_argument("--hw_size", type=int, default=144, help="Height and Width of each Training Patch (px)")
    parser.add_argument("--d_size", type=int, default=40, help="Depth of each Training Patch (px)")
    parser.add_argument("--predict_hw_size", type=int, default=96, help="Height and Width of each Patch (px) during prediction")
    parser.add_argument("--predict_depth_size", type=int, default=32, help="Depth of each Patch (px) during prediction")
    parser.add_argument("--predict_hw_overlap", type=int, default=24,
                        help="Expansion in Height and Width for each Patch (px) during prediction")
    parser.add_argument("--watershed_dynamic", type=int, default=10,
                        help="Dynamic of intensity for the search of regional minima in the distance transform image. Increasing its value will yield more object merges.")
    parser.add_argument("--predict_depth_overlap", type=int, default=4, help="Expansion in Depth for each Patch (px) during prediction")
    parser.add_argument("--result_folder_path", type=str, default="Datasets/result", help="Result Folder Path")
    parser.add_argument("--mid_visualization", action="store_false", help="Store False, so this will disable Mid Visualization")
    parser.add_argument("--train_offload", action="store_true", help="Enable disk offloading of training data")
    parser.add_argument("--val_offload", action="store_true", help="Enable disk offloading of validation data")
    parser.add_argument("--test_offload", action="store_true", help="Enable disk offloading of test data")
    parser.add_argument("--predict_offload", action="store_true", help="Enable disk offloading of prediction data")
    parser.add_argument("--model_architecture", type=str, default="SwishTransformer",
                        help="Model Architecture")
    parser.add_argument("--model_depth_multiplier", type=int, default=1, help="Model Depth multiplier")
    parser.add_argument("--model_patch_size_xy", type=int, default=6, help="Patch Height and Width (px)")
    parser.add_argument("--model_patch_size_z", type=int, default=2, help="Patch Depth (px)")
    #parser.add_argument("--find_max_channel_count", action="store_true", help="Automatically find the max channel count that won't result in an OOM error")
    parser.add_argument("--model_depth", type=int, default=8, help="Number of Transformer blocks in the Model")
    parser.add_argument("--z_to_xy_ratio", type=float, default=4.0)
    parser.add_argument("--train_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"],
                        default="Fully Labelled", help="Dataset Mode")
    #parser.add_argument("--exclude_edge", action="store_true", help="Mark pictures at object borders as unlabelled")
    #parser.add_argument("--exclude_edge_size_in", type=int, default=1, help="Pixels to exclude (inward)")
    #parser.add_argument("--exclude_edge_size_out", type=int, default=1, help="Pixels to exclude (outward)")
    parser.add_argument("--contour_map_width", type=int, default=1, help="Width of contour (outward)")
    parser.add_argument("--val_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"], default="Fully Labelled",
                        help="Dataset Mode")
    parser.add_argument("--test_dataset_mode", choices=["Fully Labelled", "Sparsely Labelled"],
                        default="Fully Labelled", help="Dataset Mode")
    #parser.add_argument("--TTA_xy", action="store_true", help="Enable Test-Time Augmentation for xy dimension")
    parser.add_argument("--instance_seg_mode", action="store_true",
                        help="Use a slower and more memory intensive watershed method for separate touching objects, "
                             "instead of simple connected component labelling. "
                             "Will yield result with much less under-segment objects.")
    parser.add_argument("--pixel_reclaim", action="store_true", help="Enable reclaim of lost pixel during the instance segmentation.")

    args = parser.parse_args()
    start_work_flow(args)
