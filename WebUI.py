import os
import math

import torch
import imageio
import gradio as gr
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipsoid_stats

from Components import DataComponents
from Components import Metrics
from tkinter import filedialog
from Networks import *
from Visualise_Network import V_N_PLModule
from read_tensorboard import read_all_tensorboard_event_files, write_to_excel
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
from command_executor import CommandExecutor

document_symbol = '\U0001F4C4'

command_executor = CommandExecutor()


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


def count_num_image_files(dataset_path, exclude_labels=True):
    count = 0
    files = os.listdir(dataset_path)
    for fname in sorted(files):
        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
            count += 1
            if exclude_labels and "Labels_" in fname:
                count -= 1
    return count


def start_work_flow(inputs):
    workflow_box_n = " ".join(inputs[workflow_box])
    existing_model_path_n = f'"{inputs[existing_model_path]}"' if inputs[existing_model_path] else ""
    train_dataset_mode_n = f'"{inputs[train_dataset_mode]}"'
    val_dataset_mode_n = f'"{inputs[val_dataset_mode]}"'
    test_dataset_mode_n = f'"{inputs[test_dataset_mode]}"'
    augmentation_csv_path_n = f'"{inputs[augmentation_csv_path]}"'
    cmd = (
           f"python workflow.py "
           f"--workflow_box {workflow_box_n} --segmentation_mode {inputs[segmentation_mode]} "
           f"--train_dataset_path {inputs[train_dataset_path]} "
           f"--augmentation_csv_path {augmentation_csv_path_n} --train_multiplier {inputs[train_multiplier]} "
           f"--batch_size {inputs[batch_size]} --num_epochs {inputs[num_epochs]} "
           f"--unsupervised_train_dataset_path {inputs[unsupervised_train_dataset_path]} "
           f"--unsupervised_train_multiplier {inputs[unsupervised_train_multiplier]} "
           f"--tensorboard_path {inputs[tensorboard_path]} "
           f"--val_dataset_path {inputs[val_dataset_path]} --test_dataset_path {inputs[test_dataset_path]} "
           f"--predict_dataset_path {inputs[predict_dataset_path]} "
           f"--existing_model_path {existing_model_path_n} "
           f"--precision {inputs[precision]} "
           f"--save_model_name {inputs[save_model_name]} --save_model_path {inputs[save_model_path]} "
           f"--train_key_name {inputs[train_key_name]} --val_key_name {inputs[val_key_name]} "
           f"--test_key_name {inputs[test_key_name]} --predict_key_name {inputs[predict_key_name]} "
           f"--hw_size {inputs[hw_size]} --d_size {inputs[d_size]} "
           f"--predict_hw_size {inputs[predict_hw_size]} --predict_depth_size {inputs[predict_depth_size]} "
           f"--predict_hw_overlap {inputs[predict_hw_overlap]} --predict_depth_overlap {inputs[predict_depth_overlap]} "
           f"--result_folder_path {inputs[result_folder_path]} "
           f"--mid_visualization_input {inputs[mid_visualization_input]} "
           f"--model_architecture {inputs[model_architecture]} --model_channel_count {inputs[model_channel_count]} "
           f"--model_depth {inputs[model_depth]} --z_to_xy_ratio {inputs[z_to_xy_ratio]} "
           f"--train_dataset_mode {train_dataset_mode_n} "
           f"--exclude_edge_size_in {inputs[exclude_edge_size_in]} --exclude_edge_size_out {inputs[exclude_edge_size_out]} "
           f"--contour_map_width {inputs[contour_map_width]} "
           f"--val_dataset_mode {val_dataset_mode_n} --test_dataset_mode {test_dataset_mode_n} ")
    if inputs[pairing_samples]:
        cmd += " --pairing_samples "
    if inputs[enable_tensorboard]:
        cmd += "--enable_tensorboard "
    if inputs[enable_unsupervised]:
        cmd += "--enable_unsupervised "
    if inputs[memory_saving_mode]:
        cmd += "--memory_saving_mode "
    if inputs[read_existing_model]:
        cmd += "--read_existing_model "
    if inputs[exclude_edge]:
        cmd += "--exclude_edge "
    if inputs[enable_mid_visualization]:
        cmd += "--enable_mid_visualization "
    if inputs[TTA_xy]:
        cmd += "--TTA_xy "
    if inputs[model_se]:
        cmd += "--model_se "
    if inputs[find_max_channel_count]:
        cmd += "--find_max_channel_count "
    if inputs[pixel_reclaim]:
        cmd += "--pixel_reclaim "

    command_executor.execute_command(cmd)


def visualisation_activations(existing_model_path, example_image, slice_to_show):
    model = V_N_PLModule.load_from_checkpoint(existing_model_path).to('cpu')
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
            tensor_depth = activation.shape[-3]
            if tensor_depth <= slice_to_show:
                activation = activation[:, :, 0:1, :, :]
            else:
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
    plt.suptitle(f'Semantic Output Layer')
    sigmoids = model.p_outs
    avg_sigmoid = torch.stack(sigmoids).mean(dim=0)
    avg_sigmoid = avg_sigmoid[:, :, slice_to_show:slice_to_show+1, :, :].squeeze()
    plt.imshow(avg_sigmoid.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img_buffer = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    figure_list.append(img_buffer)

    if len(model.c_outs) != 0:
        plt.figure(figsize=(16, 9))
        plt.suptitle(f'Contour Output Layer')
        output = model.c_outs[-1][:, :, slice_to_show:slice_to_show + 1, :, :].squeeze()
        plt.imshow(output.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
        plt.colorbar()
        plt.axis('off')
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img_buffer = np.array(canvas.renderer.buffer_rgba())
        plt.close()
        figure_list.append(img_buffer)
    return figure_list


def visualise_augmentations(train_dataset_path, hw_size, d_size, augmentation_csv,
                            slice_to_show=1, num_copies=6, pairing_samples=False, size=(8, 4.5)):
    if pairing_samples:
        datasets = ((DataComponents.TrainDataset(train_dataset_path, augmentation_csv, 1,
                                                  hw_size, d_size, negative_control='positive'), ' - positive'),
                    (DataComponents.TrainDataset(train_dataset_path, augmentation_csv, 1,
                                                hw_size, d_size, negative_control='negative'), ' - negative'))
    else:
        datasets = ((DataComponents.TrainDataset(train_dataset_path, augmentation_csv, 1, hw_size, d_size), ''),)
    figure_list = []
    for dataset in datasets:
        type = dataset[1]
        dataset = dataset[0]
        num_data = len(dataset.img_tensors)
        for i in range(0, num_data):
            # 800 x 450
            plt.figure(figsize=size)
            image_name = dataset.file_list[i][0]
            plt.suptitle(f'{image_name + type}')
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


available_architectures_semantic = [#'HalfUNetBasic',
                                    #'HalfUNetGhost',
                                    #'HalfUNetResidual',
                                    #'HalfUNetResidualBottleneck',
                                    'UNetBasic',
                                    'UNetResidual_Recommended',
                                    'UNetResidualBottleneck',
                                    #'SegNet',
                                    #'Tiniest'
                                    #'SingleTopLayer'
                                    ]
available_architectures_instance = ['InstanceBasic',
                                    'InstanceResidual_Recommended',
                                    'InstanceResidualBottleneck',]


def update_available_arch(radio_value):
    if radio_value == 'Semantic':
        return gr.Dropdown(available_architectures_semantic, label="Model Architecture", value="UNetResidual_Recommended")
    else:
        return gr.Dropdown(available_architectures_instance, label="Model Architecture", value="InstanceResidual_Recommended")


def pick_arch(arch, base_channels, depth, z_to_xy_ratio, se, unsupervised):
    model_classes = {
        "HalfUNetBasic": (Semantic_HalfUNets.HalfUNet, 'Basic'),
        "HalfUNetGhost": (Semantic_HalfUNets.HalfUNet, 'Ghost'),
        "HalfUNetResidual": (Semantic_HalfUNets.HalfUNet, 'Residual'),
        "HalfUNetResidualBottleneck": (Semantic_HalfUNets.HalfUNet, 'ResidualBottleneck'),
        "UNetBasic": (Semantic_General.UNet, 'Basic'),
        "UNetResidual_Recommended": (Semantic_General.UNet, 'Residual'),
        "UNetResidualBottleneck": (Semantic_General.UNet, 'ResidualBottleneck'),
        "SegNet": (Semantic_SegNets.Auto, 'Auto'),
        "InstanceBasic": (Instance_General.UNet, 'Basic'),
        "InstanceResidual_Recommended": (Instance_General.UNet, 'Residual'),
        "InstanceResidualBottleneck": (Instance_General.UNet, 'ResidualBottleneck')
    }

    model_class, model_type = model_classes[arch]
    return model_class(base_channels, depth, z_to_xy_ratio, model_type, se, unsupervised)


def get_stats_between_maps(predicted_path, groundtruth_path):
    ground_truth = DataComponents.path_to_tensor(groundtruth_path, label=True)
    predicted = DataComponents.path_to_tensor(predicted_path, label=True)
    ground_truth, predicted = torch.clamp(ground_truth, 0, 1), torch.clamp(predicted, 0, 1)
    intersection = 2 * torch.sum(ground_truth * predicted) + 0.001
    union = torch.sum(predicted) + torch.sum(ground_truth) + 0.001
    tp = (predicted*ground_truth).sum()
    fn = ((1-predicted)*ground_truth).sum()
    tn = ((1-predicted)*(1-ground_truth)).sum()
    fp = (predicted*(1-ground_truth)).sum()
    dice = intersection/union
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    return dice.item(), sensitivity.item(), specificity.item(), fpr.item(), fnr.item()


if __name__ == "__main__":
    with gr.Blocks(title=f"Volume Seg Tool", theme=gr.themes.Base()) as WebUI:
        with gr.Tab("Main"):
            segmentation_mode = gr.Radio(["Semantic", "Instance"], value="Semantic", label="Segmentation Mode")
            workflow_box = gr.CheckboxGroup(["Training", "Validation", "Test", "Predict"],
                                            label="Workflows to enable")
            with gr.Accordion("Workflows Explanation", open=False):
                gr.Markdown("Training: Train a network using the training dataset.")
                gr.Markdown("Validation: Use a Validation Dataset to evaluate the segmentation quality during training.")
                gr.Markdown("Test: Similar to Validation, but only run once after the model has finished training.")
                gr.Markdown("Predict: Use a (trained) network to predict the label of the predict dataset.")
                gr.Markdown("Note that if the validation is not enabled, the script will use train dataset loss to determine when to reduce the learning rate.")
            with gr.Tab("Dataset Information"):
                gr.Markdown("Necessary questions regarding the dataset in order to compute some hyperparameters.")
                gr.Markdown("Should always be filled.")
                with gr.Row():
                    question1 = gr.Number(32,
                                          label="Size to spot feature",
                                          info="Given a square shaped 2d patch randomly taken from your dataset,"
                                                " what's the minimum side length (in pixels) you would need to tell"
                                                " and segment the cellular structure of interest from the patch? "
                                                "\nIs used to compute the network's patch size, aka field of view.",
                                          precision=0)
                    z_to_xy_ratio = gr.Number(1.0, label="Z-resolution to XY-resolution ratio",
                                              info="The ratio of the z-resolution of the images in the dataset to their xy-resolution. "
                                                   "We assume xy has the same resolution.")
                    manual_mode = gr.Checkbox(scale=0,
                                              label="Check to allow manual editing of the parameters below, "
                                                    "instead of being calculated automatically")
                with gr.Accordion("Example of various field of view", open=False):
                    with gr.Row():
                        gr.Image("GitHub_Res/roi too small.png", image_mode="L", show_download_button=False,
                                 interactive=False, label="Field of view too small, cannot tell the feature")
                        gr.Image("GitHub_Res/roi just right.png", image_mode="L", show_download_button=False,
                                 interactive=False, label="Field of view just right")
                        gr.Image("GitHub_Res/roi too big.png", image_mode="L", show_download_button=False,
                                 interactive=False, label="Field of view too big, unnecessary computation and memory use")
                with gr.Accordion("Automatically computed hyperparameters", open=False):
                    with gr.Row():
                        hw_size = gr.Number(48, label="Patch Height and Width (px)", interactive=False, precision=0)
                        model_depth = gr.Number(4, label="Model Depth", interactive=False, precision=0)
                        d_size = gr.Number(48, label="Patch Depth (px)", interactive=False, precision=0)
                        dk = gr.Number(0, label="floor(log2([Z to XY ratio]))", interactive=False, precision=0)

                    def change_dataset_info_mode(choice):
                        interactive = True if choice else False
                        options = (gr.Number(48, label="Patch Height and Width (px)", interactive=interactive, precision=0),
                                   gr.Number(4, label="Model Depth", interactive=interactive, precision=0),
                                   gr.Number(48, label="Patch Depth (px)", interactive=interactive, precision=0))
                        return options
                    manual_mode.change(fn=change_dataset_info_mode, inputs=segmentation_mode, outputs=[hw_size, model_depth, d_size])

                    def calculate_dhw_size(question1, z_to_xy_ratio):
                        hw_estimate = 1.5 * question1
                        model_depth = max(round(math.log2(hw_estimate) - 3), 3)
                        hw_precision = 2 ** (model_depth - 1)
                        hw_size = hw_precision * max(round(hw_estimate / hw_precision), 1)

                        dk = math.floor(math.log2(z_to_xy_ratio))
                        d_precision = round(2 ** (model_depth - 1 - dk))
                        d_size = d_precision * max(round((hw_estimate / z_to_xy_ratio) / d_precision), 1)
                        return hw_size, model_depth, d_size, dk
                    question1.change(calculate_dhw_size, inputs=[question1, z_to_xy_ratio], outputs=[hw_size, model_depth, d_size, dk])
                    z_to_xy_ratio.change(calculate_dhw_size, inputs=[question1, z_to_xy_ratio], outputs=[hw_size, model_depth, d_size, dk])
                with gr.Row():
                    enable_unsupervised = gr.Checkbox(scale=0, label="Enable Unsupervised Learning",
                                                      info="Allow using unlabelled data to enhance performance.")
                    unsupervised_train_dataset_path = gr.Textbox('Datasets/unsupervised_train', scale=2,
                                                                 label="Unsupervised Train Dataset Path", visible=False)
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=unsupervised_train_dataset_path)
                    num_u_files = gr.Number(None,
                                            label="Number of image files in the unsupervised training set.",
                                            interactive=False, visible=False)
                    unsupervised_train_multiplier = gr.Number(None, label="Unsupervised Train Multiplier (Repeats)",
                                                              interactive=False, visible=False)

                    def show_hide_unsupervised_folder(enable_unsupervised):
                        visible = True if enable_unsupervised else False
                        options = (gr.Textbox('Datasets/unsupervised_train', scale=2,
                                   label="Unsupervised Train Dataset Path", visible=visible),
                                   gr.Number(None, label="Unsupervised Train Multiplier (Repeats)",
                                             interactive=False, visible=visible))
                        return options
                    enable_unsupervised.change(show_hide_unsupervised_folder,
                                               inputs=enable_unsupervised,
                                               outputs=[unsupervised_train_dataset_path, unsupervised_train_multiplier])

            with gr.Tab("Validation Settings"):
                with gr.Row():
                    val_key_name = gr.Textbox('Default', label="hdf5 dataset name",
                                              info='If you are using hdf5 type image (instead of tif), you need to provide its '
                                                   'dataset name.')
                with gr.Row():
                    val_dataset_path = gr.Textbox('Datasets/val', scale=2, label="Validation Dataset Path")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=val_dataset_path)

                    def calculate_val_num_patch(val_dataset_path, hw_size, d_size):
                        file_list = DataComponents.make_dataset_tv(val_dataset_path)
                        counter = 0
                        for file in file_list:
                            file = file[0]
                            depth, height, width = imageio.v3.imread(file).shape
                            depth_multiplier = max(math.ceil(depth / d_size), 1)
                            height_multiplier = max(math.ceil(height / hw_size), 1)
                            width_multiplier = max(math.ceil(width / hw_size), 1)
                            total = depth_multiplier * height_multiplier * width_multiplier
                            counter += total
                        return counter
                    val_num_patch = gr.Number(None,
                                              label="Number of patches in validation set, automatically computed",
                                              precision=0, minimum=0, interactive=False)
                val_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                            label="Dataset Mode")
            with gr.Tab("Training Settings"):
                with gr.Row():
                    train_key_name = gr.Textbox('Default', label="hdf5 dataset name",
                                                info='If you are using hdf5 type image (instead of tif), you need to provide its '
                                                     'dataset name.')
                with gr.Row():
                    train_dataset_path = gr.Textbox('Datasets/train', scale=2, label="Train Dataset Path")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=train_dataset_path)
                with gr.Row():
                    batch_size = gr.Number(1, label="Batch Size", precision=0, minimum=1, interactive=False,
                                           info="Number of training patch to feed into the network at once.")
                    pairing_samples = gr.Checkbox(label="Pairing positive and negative samples",
                                                  info="Tick this if your training data contain large area without any foreground object.")
                    def change_batch_size(choice):
                        if choice: bs = 2
                        else: bs = 1
                        return gr.Number(bs, label="Batch Size", precision=0, minimum=1, interactive=False,
                                         info="Number of training patch to feed into the network at once.")
                    pairing_samples.change(change_batch_size, inputs=pairing_samples, outputs=batch_size)
                    num_t_files = gr.Number(None,
                                            label="Number of image files in the training set.",
                                            interactive=False, visible=False)
                with gr.Accordion("When do you need to pair positive and negative sample", open=False):
                    with gr.Row():
                        gr.Image("GitHub_Res/require bs 2.png", image_mode="L", show_download_button=False,
                                 interactive=False, label="Require the box to be ticked due to large area of empty space outside cell")
                        gr.Image("GitHub_Res/do not require bs 2.png", image_mode="L", show_download_button=False,
                                 interactive=False, label="Does not Require the box to be ticked")
                with gr.Row():
                    train_multiplier = gr.Number(None, label="Train Multiplier (Repeats)",
                                                 info="Automatically calculated to aim for 10% steps in validation, 90% in train. "
                                                      "Limit to a max of 64.",
                                                 interactive=False)
                with gr.Row():
                    train_length = gr.Radio(["Short", "Medium", "Long", "Custom"], value="Short",
                                            label="Training Duration")
                    train_steps = gr.Number(2000, label="Training Steps", interactive=False)

                    def length_to_steps(train_length):
                        if train_length == "Short":
                            return gr.Number(2000, label="Training Steps", interactive=False)
                        if train_length == "Medium":
                            return gr.Number(5000, label="Training Steps", interactive=False)
                        if train_length == "Long":
                            return gr.Number(12000, label="Training Steps", interactive=False)
                        if train_length == "Custom":
                            return gr.Number(1, label="Training Steps", interactive=True, precision=0, minimum=1)
                    train_length.change(length_to_steps, inputs=train_length, outputs=train_steps)
                    num_epochs = gr.Number(None, label="Maximum Number of Epochs, automatically calculated", precision=0, minimum=1)

                    def steps_to_epochs(train_steps, train_multiplier, num_t_files):
                        return math.ceil(train_steps/(train_multiplier*num_t_files))

                with gr.Row():
                    enable_tensorboard = gr.Checkbox(True, scale=0, label="Enable TensorBoard Logging", visible=False)
                    tensorboard_path = gr.Textbox('lightning_logs', scale=2, label="Path to the folder which the log will be save to")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=tensorboard_path)
                train_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                              label="Dataset Mode")
                with gr.Row():
                    exclude_edge = gr.Checkbox(scale=0, label="Mark pixels at object borders as unlabelled", visible=True)
                    exclude_edge_size_in = gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=True, minimum=0)
                    exclude_edge_size_out = gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=True, minimum=0)
                with gr.Row():
                    contour_map_width = gr.Number(1, label="Width of contour (outward)", precision=0, visible=False, minimum=1)

                def change_edge_exclude(mode_box, train_dataset_mode):
                    show = (gr.Checkbox(scale=0,
                                        label="Mark pixels at object borders as unlabelled",
                                        visible=True),
                            gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=True,
                                      minimum=0),
                            gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=True,
                                      minimum=0))
                    no_show = (gr.Checkbox(scale=0,
                                           label="Mark pixels at object borders as unlabelled",
                                           visible=False),
                               gr.Number(1, label="Pixels to exclude (inward)", precision=0, visible=False,
                                         minimum=0),
                               gr.Number(1, label="Pixels to exclude (outward)", precision=0, visible=False,
                                         minimum=0))
                    if mode_box == "Semantic":
                        if train_dataset_mode == "Fully Labelled":
                            return show
                        else:
                            return no_show
                    else:
                        return no_show
                segmentation_mode.change(fn=change_edge_exclude, inputs=[segmentation_mode, train_dataset_mode],
                                         outputs=[exclude_edge, exclude_edge_size_in, exclude_edge_size_out])
                train_dataset_mode.change(fn=change_edge_exclude, inputs=[segmentation_mode, train_dataset_mode],
                                          outputs=[exclude_edge, exclude_edge_size_in, exclude_edge_size_out])

                def change_contour_map_width_value(choice):
                    if choice == "Instance":
                        return gr.Number(1, label="Width of contour (outward)", precision=0, visible=True,
                                         minimum=1)
                    else:
                        return gr.Number(1, label="Width of contour (outward)", precision=0, visible=False,
                                         minimum=1)
                segmentation_mode.change(fn=change_contour_map_width_value, inputs=segmentation_mode, outputs=contour_map_width)
                with gr.Row():
                    augmentation_csv_path = gr.Textbox('Augmentation Parameters.csv', scale=2,
                                                       label="Csv File for Data Augmentation Settings")
                    file_button = gr.Button(document_symbol, scale=0)
                    file_button.click(open_file, outputs=augmentation_csv_path)
                with gr.Row():
                    outputs = gr.Gallery(label="Output Images", format="png", preview=True, selected_index=0)
                    slice_to_show = gr.Number(0, visible=False)
                    number_copies = gr.Number(6, visible=False)
                    start_button = gr.Button("Show some example patches from your training dataset, "
                                             "under the current patch size and augmentation settings", scale=0)
                    start_button.click(visualise_augmentations,
                                       inputs=[train_dataset_path, hw_size, d_size, augmentation_csv_path, slice_to_show,
                                               number_copies, pairing_samples],
                                       outputs=outputs)
            with gr.Tab("Test Settings"):
                gr.Markdown("Due to the limitation of the built-in test function. The dice score obtained via the "
                            "test workflow may not represent the actual overall dice score.")
                gr.Markdown("If you want to obtain the actual dice score, it's recommended to process the image via "
                            "the Predict workflow. Then use the 'Extras' tab above to extract the overall dice score.")
                with gr.Row():
                    test_key_name = gr.Textbox('Default', label="hdf5 dataset name",
                                               info='If you are using hdf5 type image (instead of tif), you need to provide its '
                                                    'dataset name.')
                with gr.Row():
                    test_dataset_path = gr.Textbox('Datasets/test', scale=2, label="Test Dataset Path")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=test_dataset_path)
                test_dataset_mode = gr.Radio(["Fully Labelled", "Sparsely Labelled"], value="Fully Labelled",
                                             label="Dataset Mode")
            with gr.Tab("Predict Settings"):
                with gr.Row():
                    predict_key_name = gr.Textbox('Default', label="hdf5 dataset name",
                                                  info='If you are using hdf5 type image (instead of tif), you need to provide its '
                                                       'dataset name.')
                with gr.Row():
                    predict_dataset_path = gr.Textbox('Datasets/predict', scale=2, label="Predict Dataset Path")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=predict_dataset_path)
                with gr.Row():
                    predict_hw_overlap = gr.Number(2**(model_depth.value-1)/2,
                                                   label="Expansion in Height and Width for each Patch (px)",
                                                   precision=0, interactive=False)
                    predict_depth_overlap = gr.Number(2**(model_depth.value-1-dk.value)/2,
                                                      label="Expansion in Depth for each Patch (px)",
                                                      precision=0, interactive=False)
                    predict_hw_size = gr.Number(hw_size.value-2*predict_hw_overlap.value,
                                                label="Height and Width of each Patch (px)", precision=0, interactive=False)
                    predict_depth_size = gr.Number(d_size.value-2*predict_depth_overlap.value,
                                                   label="Depth of each Patch (px)", precision=0, interactive=False)

                    def calculate_predict_parameters(model_depth, hw_size, d_size, dk):
                        if hw_size <= 64:
                            hw_overlap_multiplier = 1
                        elif hw_size <= 192:
                            hw_overlap_multiplier = 2
                        else:
                            hw_overlap_multiplier = 3
                        if d_size <= 64:
                            d_overlap_multiplier = 1
                        else:
                            d_overlap_multiplier = 2
                        predict_hw_overlap = (2**(model_depth-1)/2) * hw_overlap_multiplier
                        predict_depth_overlap = (2**(model_depth-1-dk)/2) * d_overlap_multiplier
                        predict_hw_size = hw_size-2*predict_hw_overlap
                        predict_depth_size = d_size-2*predict_depth_overlap
                        options = (gr.Number(predict_hw_overlap, label="Expansion in Height and Width for each Patch (px)", interactive=False),
                                   gr.Number(predict_depth_overlap, label="Expansion in Depth for each Patch (px)", interactive=False),
                                   gr.Number(predict_hw_size, label="Height and Width of each Patch (px)", interactive=False),
                                   gr.Number(predict_depth_size, label="Depth of each Patch (px)", interactive=False))
                        return options
                    model_depth.change(calculate_predict_parameters,
                                       inputs=[model_depth, hw_size, d_size, dk],
                                       outputs=[predict_hw_overlap, predict_depth_overlap, predict_hw_size, predict_depth_size])
                    hw_size.change(calculate_predict_parameters,
                                   inputs=[model_depth, hw_size, d_size, dk],
                                   outputs=[predict_hw_overlap, predict_depth_overlap, predict_hw_size, predict_depth_size])
                    d_size.change(calculate_predict_parameters,
                                  inputs=[model_depth, hw_size, d_size, dk],
                                  outputs=[predict_hw_overlap, predict_depth_overlap, predict_hw_size, predict_depth_size])
                    dk.change(calculate_predict_parameters,
                              inputs=[model_depth, hw_size, d_size, dk],
                              outputs=[predict_hw_overlap, predict_depth_overlap, predict_hw_size, predict_depth_size])
                with gr.Row():
                    result_folder_path = gr.Textbox('Datasets/result', scale=2, label="Result Folder Path")
                    folder_button = gr.Button(document_symbol, scale=0)
                    folder_button.click(open_folder, outputs=result_folder_path)
                with gr.Row():
                    TTA_xy = gr.Checkbox(label="Enable Test-Time Augmentation for xy dimension",
                                         info="Horizontal And Vertical flip the image; the augmented images are then passed into the model."
                                              " Corresponding reverse transformation then applys to the output probability maps, and those maps get combined together."
                                              " Can improve segmentation accuracy, but will take longer and consume more CPU memory.")
                    pixel_reclaim = gr.Checkbox(label="Enable Pixel reclaim operation for instance segmentation",
                                                info="Due to how instance segmentation works, some pixels will be lost when seperating touching objects, "
                                                     "this settings will try to reclaim those lost pixels, but can take quite some time.")
                    #TTA_z = gr.Checkbox(label="Enable Test-Time Augmentation for z dimension", info="Depth Wise flip the image")
            with gr.Row():
                calculate_repeats = gr.Button(value="Calculate Training Repeats and Epoches!")
                def calculate_train_multiplier(val_num_patch, num_t_files, workflow_box):
                    if 'Validation' in workflow_box:
                        # Aiming to have 10% steps in val, 90% steps in train
                        return min(round(9 * (val_num_patch / num_t_files)), 64)
                    else:
                        return math.ceil(64 / num_t_files)

                def get_auto_parameters(workflow_box, train_dataset_path, val_dataset_path, hw_size, d_size, batch_size, train_steps, enable_unsupervised, unsupervised_train_dataset_path):
                    if 'Validation' in workflow_box:
                        val_num_patch = calculate_val_num_patch(val_dataset_path, hw_size, d_size)
                    else:
                        val_num_patch = None
                    num_t_files = count_num_image_files(train_dataset_path)
                    if enable_unsupervised:
                        num_u_files = count_num_image_files(unsupervised_train_dataset_path)
                    else:
                        num_u_files = 1
                    train_multiplier = calculate_train_multiplier(val_num_patch, num_t_files, workflow_box)
                    unsupervised_train_multiplier = (train_multiplier * num_t_files // num_u_files) * batch_size
                    num_epochs = steps_to_epochs(train_steps, train_multiplier, num_t_files)
                    return val_num_patch, num_t_files, num_u_files, train_multiplier, unsupervised_train_multiplier, num_epochs
                calculate_repeats.click(get_auto_parameters,
                                        [workflow_box, train_dataset_path, val_dataset_path, hw_size, d_size, batch_size, train_steps, enable_unsupervised, unsupervised_train_dataset_path],
                                        [val_num_patch, num_t_files, num_u_files, train_multiplier, unsupervised_train_multiplier, num_epochs])
            with gr.Row():
                read_existing_model = gr.Checkbox(label="Read Existing Model Weight File", scale=0,
                                                  info="Else it will create a new model with randomised weight.")
                existing_model_path = gr.Textbox('example_name.ckpt', label="Path to Existing Model Weight File, it should be a file with 'ckpt' at the end.")
                file_button = gr.Button(document_symbol, scale=0)
                file_button.click(open_file, outputs=existing_model_path)
            with gr.Row():
                memory_saving_mode = gr.Checkbox(label="Memory Saving Mode",
                                                 info="If you are experiencing running out of system memory (Not CUDA memory!) during training, "
                                                      "This option could help by using only 1 thread to do data loading. "
                                                      "Can significantly slow down training if your system have low single core performance.")
            with gr.Row():
                model_architecture = gr.Dropdown(available_architectures_semantic, label="Model Architecture")
                segmentation_mode.change(update_available_arch, inputs=segmentation_mode, outputs=model_architecture)
                model_channel_count = gr.Number(8, label="Base Channel Count", precision=0, minimum=1,
                                                info="Often means the number of output channels in the first encoder block. Determines the size of the network. Preferably a multiple of 8.")
                find_max_channel_count = gr.Checkbox(label="Automatically find the largest channel count",
                                                     info="Finds the largest channel count that doesn't result in a "
                                                          "Out-of-memory error through trials and errors. A slow process. "
                                                          "Certain graphic cards can be very slow with the largest "
                                                          "channel count (<0.1it/s), a lower channel count is recommended.", visible=False) # Very buggy currently...
                def show_hide_model_channel_count(find_max_channel_count):
                    if find_max_channel_count:
                        return gr.Number(8, visible=False)
                    else:
                        return gr.Number(8, label="Base Channel Count", precision=0, minimum=1, visible=True,
                                         info="Often means the number of output channels in the first encoder block. Determines the size of the network.")
                #find_max_channel_count.change(show_hide_model_channel_count, inputs=find_max_channel_count, outputs=model_channel_count)
                model_se = gr.Checkbox(True, scale=0, label="Enable Squeeze-and-Excitation plug-in",
                                       info="A simple network attention plug-in that improves segmentation accuracy at minimal cost. It is recommended to enable it.")
                def show_hide_model_tab(read_existing_model, segmentation_mode):
                    if read_existing_model:
                        visible = False
                    else:
                        visible = True
                    if segmentation_mode == 'Instance':
                        archs = available_architectures_instance
                    else:
                        archs = available_architectures_semantic
                    options = (gr.Dropdown(archs, label="Model Architecture", visible=visible),
                               gr.Number(8, label="Base Channel Count", precision=0, minimum=1,
                               info="Often means the number of output channels in the first encoder block. Determines the size of the network.", visible=visible),
                               gr.Checkbox(label="Automatically find the largest channel count",
                                           info="Finds the largest channel count that doesn't result in a "
                                                "Out-of-memory error through trials and errors. A slow process. "
                                                "Certain graphic cards can be very slow with the largest "
                                                "channel count (<0.1it/s), a lower channel count is recommended.", visible=False),
                               gr.Checkbox(True, scale=0, label="Enable Squeeze-and-Excitation plug-in",
                               info="A simple network attention plug-in that improves segmentation accuracy at minimal cost. It is recommended to enable it.", visible=visible))
                    return options

                read_existing_model.change(show_hide_model_tab, inputs=[read_existing_model, segmentation_mode], outputs=[model_architecture, model_channel_count, find_max_channel_count, model_se])
            precision = gr.Dropdown(["32", "16-mixed", "bf16-mixed"], value="32", label="Precision",
                                    info="fp16 precision could significantly cut the VRAM usage. However if you are not using an Nvidia GPU, it could signficantly slow down the training as well."
                                         "bf16 is recommended over fp16 if you are using a newer GPU (30 series or newer).")
            with gr.Accordion("Visualising training progress on the fly"):
                gr.Markdown("Note: Gradio doesn't support direct display of 3D image. The result are displayed in the tensorboard.")
                gr.Markdown("Could slow down training process, especially if the image is big.")
                enable_mid_visualization = gr.Checkbox(label="Enable Visualisation", container=False)
                with gr.Row():
                    mid_visualization_input = gr.Textbox('Datasets/mid_visualiser/image.tif', scale=1, label="Path to the input image")
                    file_button = gr.Button(document_symbol, scale=0)
                    file_button.click(open_file, outputs=mid_visualization_input)
            with gr.Row():
                save_model_name = gr.Textbox('example_name', label="File Name for Model Saved, do not include extension")
                save_model_path = gr.Textbox("'enter where you want the model to be saved'", scale=2, label="Path to Save the Model Weight",
                                             info="For path with space in it, put '' on both sides")
                folder_button = gr.Button(document_symbol, scale=0)
                folder_button.click(open_folder, outputs=save_model_path)
            input_dict = {
                workflow_box,
                segmentation_mode,
                train_dataset_path,
                augmentation_csv_path,
                train_multiplier,
                batch_size,
                pairing_samples,
                #initial_lr,
                #patience,
                #min_lr,
                num_epochs,
                enable_unsupervised,
                unsupervised_train_dataset_path,
                unsupervised_train_multiplier,
                enable_tensorboard,
                tensorboard_path,
                val_dataset_path,
                test_dataset_path,
                predict_dataset_path,
                read_existing_model,
                existing_model_path,
                precision,
                save_model_name,
                save_model_path,
                train_key_name,
                val_key_name,
                test_key_name,
                predict_key_name,
                hw_size,
                d_size,
                predict_hw_size,
                predict_depth_size,
                predict_hw_overlap,
                predict_depth_overlap,
                result_folder_path,
                enable_mid_visualization,
                mid_visualization_input,
                model_architecture,
                model_channel_count,
                find_max_channel_count,
                memory_saving_mode,
                model_depth,
                z_to_xy_ratio,
                model_se,
                train_dataset_mode,
                exclude_edge,
                exclude_edge_size_in,
                exclude_edge_size_out,
                contour_map_width,
                val_dataset_mode,
                test_dataset_mode,
                TTA_xy,
                pixel_reclaim,
#                TTA_z,
                }
            with gr.Row():
                start_button = gr.Button("Start Workflow", elem_id="start_button")
                start_button.click(start_work_flow, inputs=input_dict)
                stop_button = gr.Button("Stop Workflow", elem_id="stop_button")
                #stop_button.click(stop_training_callback)
                stop_button.click(command_executor.kill_command)

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
            slice_to_show = gr.Number(0, label="Depth Slice to show", precision=0, minimum=0)
            outputs = gr.Gallery(label="Output Images", preview=True, selected_index=0)
            start_button = gr.Button("Show Visualization")
            start_button.click(visualisation_activations, inputs=[existing_model_path_av, image_path_av, slice_to_show], outputs=outputs)

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
            with gr.Row():
                slice_to_show = gr.Number(0, label="Depth Slice to show", precision=0, minimum=0)
                num_copies = gr.Number(6, label="Number of examples for each file", precision=0, minimum=6)
            outputs = gr.Gallery(label="Output Images", format="png", preview=True, selected_index=0)
            start_button = gr.Button("Show Visualization")
            start_button.click(visualise_augmentations, inputs=[train_dataset_path_av, hw_size, d_size,
                                                                augmentation_csv_path_av, slice_to_show, num_copies], outputs=outputs)

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
                    folder_button.click(open_folder, outputs=save_log_path)
                save_button = gr.Button("Output TensorBoard log to Excel")
                save_button.click(tensorboard_to_excel, inputs=[tensorboard_path_e, save_log_name, save_log_path])
            with gr.Accordion("Calculate statistics between the predicted labels and ground truth labels"):
                with gr.Row():
                    predicted_img_path = gr.Textbox(label='Path to the predicted image')
                    file_button = gr.Button(document_symbol, scale=0)
                    file_button.click(open_file, outputs=predicted_img_path)
                with gr.Row():
                    ground_truth_img_path = gr.Textbox(label='Path to the ground truth image')
                    file_button = gr.Button(document_symbol, scale=0)
                    file_button.click(open_file, outputs=ground_truth_img_path)
                with gr.Row():
                    dice = gr.Number(interactive=False, label="Dice Score")
                    sensitivity = gr.Number(interactive=False, label="Sensitivity (True Positive Rate)")
                    specificity = gr.Number(interactive=False, label="Specificity (True Negative Rate)")
                    fpr = gr.Number(interactive=False, label="False Positive Rate")
                    fnr = gr.Number(interactive=False, label="False Negative Rate")
                start_button = gr.Button("Get statistics!")
                start_button.click(get_stats_between_maps, inputs=[predicted_img_path, ground_truth_img_path],
                                   outputs=[dice, sensitivity, specificity, fpr, fnr])

    WebUI.launch(inbrowser=True)
