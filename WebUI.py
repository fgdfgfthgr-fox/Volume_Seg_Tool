import os
import math
import time
import threading
import subprocess
import webbrowser

import lightning.pytorch as pl
import torch
import gradio as gr
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

from run_semantic import PLModuleSemantic
from run_instance import PLModuleInstance
from Components import DataComponents
from pytorch_lightning.loggers import TensorBoardLogger
from tkinter import filedialog
from Networks import *
from Visualise_Network import V_N_PLModule
from read_tensorboard import read_all_tensorboard_event_files, write_to_excel

document_symbol = '\U0001F4C4'

num_cpu_cores = os.cpu_count()
desired_num_workers = max(num_cpu_cores-1, 1)

stop_flag = False

def create_logger(inputs):
    logger = TensorBoardLogger(f'{inputs[tensorboard_path]}', name='Run')
    return logger


def start_tensorboard(inputs):
    # Read the TENSORBOARD_PORT from the environment, or use the default
    tensorboard_port = os.environ.get('TENSORBOARD_PORT', 6006)
    subprocess.run(f"tensorboard --logdir={inputs[tensorboard_path]} --port={str(tensorboard_port)}", shell=True)
    tensorboard_url = f'http://localhost:{tensorboard_port}'
    webbrowser.open(tensorboard_url, 2)


def open_folder():
    root = tk.Tk()
    #root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Display the dialog in the foreground.
    root.iconify()  # Hide the little window.
    folder_path = filedialog.askdirectory()
    root.destroy()
    return str(folder_path)


def open_file():
    root = tk.Tk()
    #root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Display the dialog in the foreground.
    root.iconify()  # Hide the little window.
    file_path = filedialog.askopenfilename()
    root.destroy()
    return str(file_path)


def start_work_flow_thread(inputs):
    global stop_event
    stop_event = threading.Event()
    thread = threading.Thread(target=start_work_flow, args=(inputs, stop_event,))
    thread.start()
    return thread


def stop_work_flow_thread(thread, stop_event):
    stop_event.set()
    thread.join()


def start_work_flow(inputs, stop_event):
    torch.set_float32_matmul_precision('medium')
    while not stop_event.is_set():
        if 'Training' in inputs[workflow_box]:
            if 'Semantic' in inputs[mode_box]:
                train_dataset = DataComponents.TrainDataset(inputs[train_dataset_path], inputs[augmentation_csv_path], inputs[train_multiplier],
                                                            inputs[exclude_edge], inputs[exclude_edge_size_in], inputs[exclude_edge_size_out])
            else:
                train_dataset = DataComponents.TrainDatasetInstance(inputs[train_dataset_path], inputs[augmentation_csv_path],
                                                                    inputs[train_multiplier])
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=inputs[batch_size],
                                                       shuffle=True, num_workers=desired_num_workers, persistent_workers=True, pin_memory=True)
            if 'Validation' in inputs[workflow_box]:
                if 'Semantic' in inputs[mode_box]:
                    val_dataset = DataComponents.ValDataset(inputs[val_dataset_path], inputs[augmentation_csv_path])
                else:
                    val_dataset = DataComponents.ValDatasetInstance(inputs[val_dataset_path], inputs[augmentation_csv_path])
                val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=inputs[batch_size],
                                                         num_workers=desired_num_workers, persistent_workers=True, pin_memory=True)
            else:
                val_loader = None
            if 'Test' in inputs[workflow_box]:
                if 'Semantic' in inputs[mode_box]:
                    test_dataset = DataComponents.ValDataset(inputs[test_dataset_path], inputs[augmentation_csv_path])
                else:
                    test_dataset = DataComponents.ValDatasetInstance(inputs[test_dataset_path], inputs[augmentation_csv_path])
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=inputs[batch_size])
        arch = pick_arch(inputs[model_architecture], inputs[model_channel_count], inputs[model_depth], inputs[model_z_to_xy_ratio])
        if ('Sparsely Labelled' in inputs[train_dataset_mode]) or (inputs[exclude_edge]):
            sparse_train = True
        else:
            sparse_train = False
        if 'Semantic' in inputs[mode_box]:
            model = PLModuleSemantic(arch,
                                     inputs[initial_lr], inputs[patience], inputs[min_lr],
                                     'Validation' in inputs[workflow_box], inputs[enable_mid_visualization],
                                     inputs[mid_visualization_input],
                                     sparse_train, 'Sparsely Labelled' in inputs[val_dataset_mode], 'Sparsely Labelled' in inputs[test_dataset_mode])
        else:
            model = PLModuleInstance(arch,
                                     inputs[initial_lr], inputs[patience], inputs[min_lr],
                                     'Validation' in inputs[workflow_box], inputs[enable_mid_visualization],
                                     inputs[mid_visualization_input],
                                     sparse_train, 'Sparsely Labelled' in inputs[val_dataset_mode],
                                     'Sparsely Labelled' in inputs[test_dataset_mode])
        if inputs[read_existing_model]:
            model.load_state_dict(torch.load(inputs[existing_model_path]))
        if inputs[enable_tensorboard]:
            tensorboard_thread = threading.Thread(target=start_tensorboard, args=[inputs])
            tensorboard_thread.daemon = True
            tensorboard_thread.start()
            logger = create_logger(inputs)
        else:
            logger = None
        if 'Training' in inputs[workflow_box]:
            trainer = pl.Trainer(max_epochs=inputs[max_epochs], log_every_n_steps=1, logger=logger,
                                 accelerator="gpu", enable_checkpointing=False,
                                 precision=inputs[precision], enable_progress_bar=True, num_sanity_val_steps=0,)
            start_time = time.time()
            trainer.fit(model,
                        val_dataloaders=val_loader,
                        train_dataloaders=train_loader)
            end_time = time.time()
            print(f"Training Taken: {end_time - start_time} seconds")
        if 'Test' in inputs[workflow_box]:
            trainer = pl.Trainer(precision=inputs[precision], enable_progress_bar=True, logger=False, accelerator="gpu")
            trainer.test(model, dataloaders=test_loader)
        if inputs[save_model]:
            full = os.path.join(inputs[save_model_path], inputs[save_model_name])
            torch.save(model.state_dict(), full)
        if 'Predict' in inputs[workflow_box]:
            trainer = pl.Trainer(precision=inputs[precision], enable_progress_bar=True, logger=False, accelerator="gpu")
            predict_dataset = DataComponents.Predict_Dataset(inputs[predict_dataset_path],
                                                             hw_size=inputs[predict_hw_size], depth_size=inputs[predict_depth_size],
                                                             hw_overlap=inputs[predict_hw_overlap], depth_overlap=inputs[predict_depth_overlap],
                                                             TTA_hw=inputs[TTA_xy])
            predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
            meta_info = predict_dataset.__getmetainfo__()
            start_time = time.time()
            predictions = trainer.predict(model, predict_loader)
            end_time = time.time()
            print(f"Prediction Taken: {end_time - start_time} seconds")
            if 'Semantic' in inputs[mode_box]:
                start_time = time.time()
                DataComponents.predictions_to_final_img(predictions, meta_info, direc=inputs[result_folder_path],
                                                        hw_size=inputs[predict_hw_size], depth_size=inputs[predict_depth_size],
                                                        hw_overlap=inputs[predict_hw_overlap], depth_overlap=inputs[predict_depth_overlap],
                                                        TTA_hw=inputs[TTA_xy])
                end_time = time.time()
                print(f"Converting and saving taken: {end_time - start_time} seconds")
            else:
                start_time = time.time()
                DataComponents.predictions_to_final_img_instance(predictions, meta_info, direc=inputs[result_folder_path],
                                                                 hw_size=inputs[predict_hw_size], depth_size=inputs[predict_depth_size],
                                                                 hw_overlap=inputs[predict_hw_overlap], depth_overlap=inputs[predict_depth_overlap],
                                                                 TTA_hw=inputs[TTA_xy])
                end_time = time.time()
                print(f"Converting and saving taken: {end_time - start_time} seconds")


def visualisation_activations(existing_model_path, example_image, slice_to_show,
                              model_architecture, model_channel_count, model_depth, model_z_to_xy_ratio):
    arch = pick_arch(model_architecture, model_channel_count, model_depth, model_z_to_xy_ratio)
    model = V_N_PLModule(arch)
    model.load_state_dict(torch.load(existing_model_path))
    figure_list = []
    tensor = DataComponents.path_to_tensor(example_image).unsqueeze(0).unsqueeze(0)
    # Pass the test tensor through the model
    with torch.no_grad():
        model(tensor)

    plt.figure(figsize=(16, 9))
    plt.suptitle(f'Input')
    initial = tensor[:, :, slice_to_show:slice_to_show+1, :, :].squeeze()
    plt.imshow(initial.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    # Render the figure to an image buffer and convert it to a NumPy array
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img_buffer = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    figure_list.append(img_buffer)

    for i, activation in enumerate(model.activations):
        if len(activation.shape) == 5:
            activation = activation[:, :, slice_to_show:slice_to_show+1, :, :]
            tensor_width = activation.shape[-1]
            tensor_height = activation.shape[-2]
            if tensor_width >= 4 and tensor_height >= 4:
                plt.figure(figsize=(16, 9))
                channels = torch.split(activation, 1, dim=1)
                num_channels = len(channels)
                plt.suptitle(f'Activation Layer {i}, {num_channels} channels, {tensor_width} * {tensor_height}')

                # Create a grid of subplots to display channels
                rows = math.floor(math.sqrt(num_channels))
                cols = math.ceil(num_channels/rows)

                for j, channel in enumerate(channels):
                    channel = channel.squeeze()
                    plt.subplot(rows, cols, j + 1)
                    plt.imshow(channel.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
                    plt.axis('off')

                plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
                canvas = plt.get_current_fig_manager().canvas
                canvas.draw()
                img_buffer = np.array(canvas.renderer.buffer_rgba())
                plt.close()
                figure_list.append(img_buffer)

    plt.figure(figsize=(16, 9))
    plt.suptitle(f'Sigmoid Layer')
    sigmoid = model.sigmoids[-1][:, :, slice_to_show:slice_to_show+1, :, :].squeeze()
    plt.imshow(sigmoid.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img_buffer = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    figure_list.append(img_buffer)
    return figure_list


def visualisation_augmentations(train_dataset_path, augmentation_csv, slice_to_show):
    dataset = DataComponents.TrainDataset(train_dataset_path, augmentation_csv, 1)
    num_data = len(dataset.img_tensors)
    num_copies = 12
    figure_list = []
    for i in range(0, num_data):
        # 1600 x 900
        plt.figure(figsize=(16, 9))
        image_name = dataset.file_list[i][0]
        plt.suptitle(f'{image_name}')
        rows = math.floor(math.sqrt(num_copies * 2))
        cols = math.ceil(num_copies / rows)
        for k in range(0, num_copies):
            pair = dataset.__getitem__(i)
            image = pair[0][:, slice_to_show:slice_to_show+1, :, :].squeeze()
            label = pair[1][:, slice_to_show:slice_to_show+1, :, :].squeeze()

            # Plot Image
            plt.subplot(rows, 2 * cols, 2 * k + 1)
            plt.imshow(image.cpu().numpy(), cmap='gist_gray')
            plt.axis('off')

            # Plot Label
            plt.subplot(rows, 2 * cols, 2 * k + 2)
            plt.imshow(label.cpu().numpy(), cmap='gist_gray')
            plt.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img_buffer = np.array(canvas.renderer.buffer_rgba())
        plt.close()
        figure_list.append(img_buffer)
    return figure_list


def tensorboard_to_excel(log_path, save_log_name, save_log_path):
    # Read all TensorBoard event files in the folder and merge results
    merged_data = read_all_tensorboard_event_files(log_path)

    # Write merged data to Excel file
    write_to_excel(merged_data, save_log_path+save_log_name)

    print(f"Data from all TensorBoard event files in {log_path} has been merged and written to {save_log_name}.")


available_architectures_semantic = ['HalfUNetBasic',
                                    'HalfUNetGhost',
                                    'HalfUNetResidual',
                                    'HalfUNetResidualBottleneck',
                                    'UNetBasic',
                                    'UNetResidual',
                                    'UNetResidualBottleneck',
                                    'SegNet']
available_architectures_instance = ['InstanceBasic',
                                    'InstanceResidual',
                                    'InstanceResidualBottleneck',]


def pick_arch(arch, base_channels, depth, z_to_xy_ratio):
    if arch == "HalfUNetBasic":
        return Semantic_HalfUNets.Basic(base_channels, depth, z_to_xy_ratio)
    elif arch == "HalfUNetGhost":
        return Semantic_HalfUNets.Ghost(base_channels, depth, z_to_xy_ratio)
    elif arch == "HalfUNetResidual":
        return Semantic_HalfUNets.Residual(base_channels, depth, z_to_xy_ratio)
    elif arch == "HalfUNetResidualBottleneck":
        return Semantic_HalfUNets.ResidualBottleneck(base_channels, depth, z_to_xy_ratio)
    elif arch == "UNetBasic":
        return Semantic_General.Basic(base_channels, depth, z_to_xy_ratio)
    elif arch == "UNetResidual":
        return Semantic_General.Residual(base_channels, depth, z_to_xy_ratio)
    elif arch == "SegNet":
        return Semantic_SegNets.Auto(base_channels, depth, z_to_xy_ratio)
    elif arch == "UNetResidualBottleneck":
        return Semantic_General.ResidualBottleneck(base_channels, depth, z_to_xy_ratio)
    elif arch == "InstanceBasic":
        return Instance_General.Basic(base_channels, depth, z_to_xy_ratio)
    elif arch == "InstanceResidual":
        return Instance_General.Residual(base_channels, depth, z_to_xy_ratio)
    elif arch == "InstanceResidualBottleneck":
        return Instance_General.ResidualBottleneck(base_channels, depth, z_to_xy_ratio)


def change_edge_exclude(choice):
    if choice == "Fully Labelled":
        return gr.Checkbox(scale=0, label="Exclude object borders, do not work in Instance Segmentation Mode", visible=True)
    else:
        return gr.Checkbox(scale=0, label="Exclude object borders, do not work in Instance Segmentation Mode", visible=False)


def change_edge_exclude_value_in(choice):
    if choice == "Fully Labelled":
        return gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=True)
    else:
        return gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=False)


def change_edge_exclude_value_out(choice):
    if choice == "Fully Labelled":
        return gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=True)
    else:
        return gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=False)


def update_available_arch(radio_value):
    if radio_value == 'Semantic':
        return gr.Dropdown(available_architectures_semantic, label="Model Architecture")
    else:
        return gr.Dropdown(available_architectures_instance, label="Model Architecture")


if __name__ == "__main__":
    # Function to start workflow when the GUI button is pressed
    def start_training_callback(inputs):
        global workflow_thread
        workflow_thread = start_work_flow_thread(inputs)

    # Function to stop workflow when the GUI button is pressed
    def stop_training_callback():
        stop_work_flow_thread(workflow_thread, stop_event)

    with gr.Blocks(title=f"Volume Seg Tool") as WebUI:
        with gr.Tab("Main"):
            mode_box = gr.Radio(["Semantic", "Instance (BETA)"], value="Semantic", label="Segmentation Mode")
            workflow_box = gr.CheckboxGroup(["Training", "Validation", "Test", "Predict"],
                                            label="Workflows to enable")
            with gr.Accordion("Workflows Explanation", open=False):
                gr.Markdown("Training: Train a network using the training dataset.")
                gr.Markdown("Validation: Use a Validation Dataset to evaluate the network during training.")
                gr.Markdown("Test: Similar to Validation, but only run once after the model has finished training.")
                gr.Markdown("Predict: Use a (trained) network to predict the label of the predict dataset.")
                gr.Markdown("Note that if the validation is not enabled, the script will use train dataset loss to determine when to reduce the learning rate.")
            with gr.Row():
                with gr.Tab("Training Settings"):
                    with gr.Row():
                        train_dataset_path = gr.Textbox('Datasets/train', scale=2, label="Train Dataset Path")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=train_dataset_path)
                    with gr.Row():
                        batch_size = gr.Number(1, label="Batch Size", precision=0)
                        initial_lr = gr.Number(0.001, step=0.001, label="Initial Learning Rate")
                        patience = gr.Number(5, label="Learning Rate Decay Patience", precision=0)
                        min_lr = gr.Number(0.0001, step=0.0001, label="Minimum Learning Rate")
                        max_epochs = gr.Number(40, label="Maximum Number of Epochs", precision=0)
                        train_multiplier = gr.Number(8, label="Train Multiplier (Repeats)", precision=0)
                    with gr.Row():
                        enable_tensorboard = gr.Checkbox(scale=0, label="Enable TensorBoard Logging")
                        tensorboard_path = gr.Textbox('lightning_logs', scale=2, label="Path to the folder which the log will be save to")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=tensorboard_path)
                    train_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                                  label="Dataset Mode")
                    with gr.Row():
                        exclude_edge = gr.Checkbox(scale=0, label="Mark pictures at object borders as unlabelled, do not work in Instance Segmentation Mode", visible=True)
                        exclude_edge_size_in = gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=True)
                        exclude_edge_size_out = gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=True)
                    train_dataset_mode.change(fn=change_edge_exclude, inputs=train_dataset_mode, outputs=exclude_edge)
                    train_dataset_mode.change(fn=change_edge_exclude_value_in, inputs=train_dataset_mode, outputs=exclude_edge_size_in)
                    train_dataset_mode.change(fn=change_edge_exclude_value_out, inputs=train_dataset_mode, outputs=exclude_edge_size_out)
                with gr.Tab("Validation Settings"):
                    with gr.Row():
                        val_dataset_path = gr.Textbox('Datasets/val', scale=2, label="Validation Dataset Path")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=val_dataset_path)
                    val_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                                label="Dataset Mode")
                with gr.Tab("Test Settings"):
                    with gr.Row():
                        test_dataset_path = gr.Textbox('Datasets/test', scale=2, label="Test Dataset Path")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=test_dataset_path)
                    test_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                                 label="Dataset Mode")
                with gr.Tab("Predict Settings"):
                    with gr.Row():
                        predict_dataset_path = gr.Textbox('Datasets/predict', scale=2, label="Predict Dataset Path")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=predict_dataset_path)
                    with gr.Row():
                        predict_hw_size = gr.Number(128, label="Height and Width of each Patch (px)", precision=0, step=16)
                        predict_depth_size = gr.Number(128, label="Depth of each Patch (px)", precision=0, step=16)
                        predict_hw_overlap = gr.Number(8, label="Expansion in Height and Width for each Patch (px)", precision=0, step=8)
                        predict_depth_overlap = gr.Number(8, label="Expansion in Depth for each Patch (px)", precision=0, step=8)
                    with gr.Row():
                        result_folder_path = gr.Textbox('Datasets/result', scale=2, label="Result Folder Path")
                        folder_button = gr.Button(document_symbol, scale=0)
                        folder_button.click(open_folder, outputs=result_folder_path)
                    with gr.Row():
                        TTA_xy = gr.Checkbox(label="Enable Test-Time Augmentation for xy dimension", info="Horizontal And Vertical flip the image; the augmented images are then passed into the model. The output probability maps are applied via the corresponding reverse transformation and combined.")
                        #TTA_z = gr.Checkbox(label="Enable Test-Time Augmentation for z dimension", info="Depth Wise flip the image")
            with gr.Row():
                model_architecture = gr.Dropdown(available_architectures_semantic, label="Model Architecture")
                mode_box.change(update_available_arch, inputs=mode_box, outputs=model_architecture)
                model_channel_count = gr.Number(8, label="Base Channel Count", precision=0,
                                                info="Often means the number of output channels in the first encoder block. Determines the size of the network.")
                model_depth = gr.Number(5, label="Model Depth", precision=0,
                                        info="Only work for HalfUNet and UNet, minimal is 3. Means the number of different downsampled sizes. Including un-downsampled.")
                model_z_to_xy_ratio = gr.Number(1.0, label="Z resolution to XY resolution ratio",
                                                info="The ratio which the z resolution of the images in the dataset divided by their xy resolution. Determines some internal model layout. We assume xy has the same resolution.")
#                model_growth_rate = gr.Number(12, label="Model Growth Rate", precision=0,
#                                              info="Only work for DenseNet, check out their paper for more detail.")
#                model_dropout_p = gr.Number(0.2, label="Model Dropout Probability", step=0.1,
#                                            info="Only work for DenseNet, check out their paper for more detail.")
            with gr.Row():
                augmentation_csv_path = gr.Textbox('Augmentation Parameters.csv', scale=2, label="Csv File for Data Augmentation Settings")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=augmentation_csv_path)
            precision = gr.Dropdown(["32", "16", "bf16"], value="32", label="Precision",
                                    info="fp16 precision could significantly cut the VRAM usage. However if you are not using an Nvidia GPU, it could signficantly slow down the training as well."
                                         "bf16 is for cpu-only training, which takes less RAM than fp32.")
            with gr.Row():
                read_existing_model = gr.Checkbox(label="Read Existing Model Weight File", scale=0,
                                                  info="Else it will create a new model with randomised weight.")
                existing_model_path = gr.Textbox(label="Path to Existing Model Weight File")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=existing_model_path)
            with gr.Accordion("Visualising training progress on the fly"):
                gr.Markdown("Note: Gradio doesn't support direct display of 3D image. The result are displayed in the tensorboard.")
                gr.Markdown("Could slow down training process, especially if the image is big.")
                enable_mid_visualization = gr.Checkbox(label="Enable Visualisation", container=False)
                with gr.Row():
                    mid_visualization_input = gr.Textbox('Datasets/mid_visualiser/image.tif', scale=1, label="Path to the input image")
                    file_button = gr.Button(document_symbol, scale=0)
                    file_button.click(open_file, outputs=mid_visualization_input)
            with gr.Row():
                save_model = gr.Checkbox(label="Save Trained Model Weight")
                save_model_name = gr.Textbox('example_name.pth', label="File Name for Model Saved, including extension",
                                             info="Recommend extension: .pth")
                save_model_path = gr.Textbox(scale=2, label="Path to Save the Model Weight")
                folder_button = gr.Button(document_symbol, scale=0)
                folder_button.click(open_folder, outputs=save_model_path)
            input_dict = {
                workflow_box,
                mode_box,
                train_dataset_path,
                augmentation_csv_path,
                train_multiplier,
                batch_size,
                initial_lr,
                patience,
                min_lr,
                max_epochs,
                enable_tensorboard,
                tensorboard_path,
                val_dataset_path,
                test_dataset_path,
                predict_dataset_path,
                read_existing_model,
                existing_model_path,
                precision,
                save_model,
                save_model_name,
                save_model_path,
                predict_hw_size,
                predict_depth_size,
                predict_hw_overlap,
                predict_depth_overlap,
                result_folder_path,
                enable_mid_visualization,
                mid_visualization_input,
                model_architecture,
                model_channel_count,
                model_depth,
                model_z_to_xy_ratio,
                train_dataset_mode,
                exclude_edge,
                exclude_edge_size_in,
                exclude_edge_size_out,
                val_dataset_mode,
                test_dataset_mode,
                TTA_xy,
#                TTA_z,
                }
            with gr.Row():
                start_button = gr.Button("Start Workflow", elem_id="start_button")
                start_button.click(start_training_callback, inputs=input_dict)
                stop_button = gr.Button("Stop Workflow", elem_id="stop_button")
                stop_button.click(stop_training_callback)

        with gr.Tab("Activations Visualisation"):
            gr.Markdown("Given an example image and a trained model weight, visualize the model output in each activation layers.")
            gr.Markdown("As well as the sigmoid layer (the layer right before the output).")
            with gr.Row():
                existing_model_path_av = gr.Textbox(label="Path to the Model Weight File")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=existing_model_path_av)
            with gr.Row():
                image_path_av = gr.Textbox(label="Path to the Example Image")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=image_path_av)
            slice_to_show = gr.Number(0, label="Depth Slice to show", precision=0)
            with gr.Row():
                model_architecture_v = gr.Dropdown(available_architectures_semantic, label="Model Architecture")
                mode_box.change(update_available_arch, inputs=mode_box, outputs=model_architecture_v)
                model_channel_count_v = gr.Number(8, label="Base Channel Count", precision=0,
                                                info="Often means the number of output channels in the first encoder block. Determines the size of the network.")
                model_depth_v = gr.Number(5, label="Model Depth", precision=0,
                                        info="Only work for HalfUNet and UNet, minimal is 3. Means the number of different downsampled sizes. Including un-downsampled.")
#                model_growth_rate_v = gr.Number(12, label="Model Growth Rate", precision=0,
#                                              info="Only work for DenseNet, check out their paper for more detail.")
#                model_dropout_p_v = gr.Number(0.2, label="Model Dropout Probability", step=0.1,
#                                            info="Only work for DenseNet, check out their paper for more detail.")
                model_z_to_xy_ratio_v = gr.Number(1.0, label="Z resolution to XY resolution ratio",
                                                info="The ratio which the z resolution of the images in the dataset divided by their xy resolution. Determines some internal model layout. We assume xy has the same resolution.")
            outputs = gr.Gallery(label="Output Images", preview=True, selected_index=0)
            start_button = gr.Button("Show Visualization")
            start_button.click(visualisation_activations, inputs=[existing_model_path_av, image_path_av, slice_to_show,
                                                                  model_architecture_v, model_channel_count_v, model_depth_v,
                                                                  model_z_to_xy_ratio_v], outputs=outputs)

        with gr.Tab("Augmentations Visualisation"):
            gr.Markdown("Given your Training Dataset and Augmentation CSV, show some examples of augmented images that will be fed into the network.")
            with gr.Row():
                train_dataset_path_av = gr.Textbox('Datasets/train', label="Path to the Training Dataset")
                folder_button = gr.Button(document_symbol, scale=0)
                folder_button.click(open_folder, outputs=train_dataset_path_av)
            with gr.Row():
                augmentation_csv_path_av = gr.Textbox('Augmentation Parameters.csv', scale=2, label="Csv File for Data Augmentation Settings")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=augmentation_csv_path_av)
            slice_to_show = gr.Number(0, label="Depth Slice to show", precision=0)
            outputs = gr.Gallery(label="Output Images", preview=True, selected_index=0)
            start_button = gr.Button("Show Visualization")
            start_button.click(visualisation_augmentations, inputs=[train_dataset_path_av, augmentation_csv_path_av, slice_to_show], outputs=outputs)

        with gr.Tab("Extras"):
            with gr.Accordion("Output TensorBoard log to Excel"):
                with gr.Row():
                    tensorboard_path_e = gr.Textbox('lightning_logs', scale=2,
                                                    label="Path to the folder which the tensorboard log are saved to")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=tensorboard_path_e)
                with gr.Row():
                    save_log_name = gr.Textbox('output.xlsx',
                                               label="File name for the output Excel file, including extension")
                    save_log_path = gr.Textbox(scale=2, label="Path to Save the Excel file")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=save_model_path)
                save_button = gr.Button("Output TensorBoard log to Excel")
                save_button.click(tensorboard_to_excel, inputs=[tensorboard_path_e, save_log_name, save_log_path])
    WebUI.launch(inbrowser=True)
