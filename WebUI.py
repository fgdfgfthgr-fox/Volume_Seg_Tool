import gc
import io
import os
import sys
import json
import math
import base64
import threading

import torch
import tkinter as tk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import webview
from PIL import Image
from tkinter import filedialog

from Components import Utils, Metrics, Datasets
from read_tensorboard import read_all_tensorboard_event_files, write_to_excel
from torchvision.datasets.folder import has_file_allowed_extension
from command_executor import CommandExecutor

document_symbol = '\U0001F4C4'


# ---------------------------------------------------------------------------
# Helpers (preserved from the Gradio version)
# ---------------------------------------------------------------------------

def open_folder_dialog():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.iconify()
    folder_path = filedialog.askdirectory()
    root.destroy()
    return str(folder_path) if folder_path else ''


def open_file_dialog():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.iconify()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return str(file_path) if file_path else ''


def count_num_image_files(dataset_path, exclude_labels=True):
    count = 0
    files = os.listdir(dataset_path)
    for fname in sorted(files):
        if has_file_allowed_extension(fname, (".tif", ".tiff", ".mrc", ".h5", ".hdf5")):
            count += 1
            if exclude_labels and "Labels_" in fname:
                count -= 1
    return count


def classify_gpu_type(name):
    name_lower = name.lower()
    nvidia_keywords = ['nvidia', 'geforce', 'quadro', 'tesla', 'titan', 'rtx']
    amd_discrete_keywords = ['radeon rx', 'radeon pro', 'radeon r9', 'radeon r7',
                             'vega frontier', 'mi100', 'mi200', 'mi250', 'mi300']
    amd_integrated_keywords = ['radeon vega', 'radeon graphics', 'graphics']
    intel_keywords = ['intel', 'uhd', 'iris', 'hd graphics']

    if any(kw in name_lower for kw in nvidia_keywords):
        return 'discrete'
    if any(kw in name_lower for kw in intel_keywords):
        return 'integrated'
    if any(kw in name_lower for kw in amd_discrete_keywords):
        return 'discrete'
    if any(kw in name_lower for kw in amd_integrated_keywords):
        return 'integrated'
    return 'discrete'


def calculate_val_num_patch(val_dataset_path, hw_size, d_size, val_key_name='Default'):
    file_list = Utils.make_label_pair_tv(val_dataset_path)
    counter = 0
    for file in file_list:
        file = Utils.multiple_loader(file[0], val_key_name)
        depth, height, width = file.shape
        depth_multiplier = max(math.ceil(depth / d_size), 1)
        height_multiplier = max(math.ceil(height / hw_size), 1)
        width_multiplier = max(math.ceil(width / hw_size), 1)
        total = depth_multiplier * height_multiplier * width_multiplier
        counter += total
    return counter


def steps_to_epochs(train_steps, train_multiplier, num_t_files):
    return math.ceil(train_steps / (train_multiplier * num_t_files))


def calculate_train_multiplier(val_num_patch, num_t_files, workflow_box):
    if 'Validation' in workflow_box:
        return min(round(9 * (val_num_patch / num_t_files)), 128)
    else:
        return math.ceil(128 / num_t_files)


def visualise_augmentations(train_dataset_path, hw_size, d_size, augmentation_csv,
                            slice_to_show=1, num_copies=6, pairing_samples=False,
                            segmentation_mode='Semantic',
                            contour_map_width=1, train_key_name='Default'):
    if segmentation_mode == 'Semantic':
        instance_mode = False
        img_each_col = 2
    else:
        instance_mode = True
        img_each_col = 3

    dataset = Datasets.TrainDataset(train_dataset_path, augmentation_csv, 1,
                                    hw_size, d_size, instance_mode,
                                    contour_map_width, train_key_name)

    if pairing_samples:
        types = [' - positive', ' - negative']
    else:
        types = ['']

    figure_list = []
    num_data = len(dataset.img_tensors)

    for type_str in types:
        if type_str == ' - positive':
            arg = 'positive'
        elif type_str == ' - negative':
            arg = 'negative'
        else:
            arg = None

        for i in range(0, num_data):
            plt.figure(figsize=(9, 4.5))
            image_name = str(dataset.file_list[i][0])
            plt.suptitle(f'{image_name}{type_str}')
            rows = math.floor(math.sqrt(num_copies * img_each_col))
            cols = math.ceil(num_copies / rows)

            for k in range(0, num_copies):
                pair = dataset.__getitem__((i, arg))
                image = pair[0][:, slice_to_show:slice_to_show + 1, :, :].squeeze()
                label = pair[1][:, slice_to_show:slice_to_show + 1, :, :].squeeze()
                contour = None
                if len(pair) == 3:
                    contour = pair[2][:, slice_to_show:slice_to_show + 1, :, :].squeeze()

                plt.subplot(rows, img_each_col * cols, img_each_col * k + 1)
                plt.imshow(image.cpu().numpy(), cmap='gist_gray')
                plt.axis('off')

                plt.subplot(rows, img_each_col * cols, img_each_col * k + 2)
                plt.imshow(label.cpu().numpy(), cmap='gist_gray')
                plt.axis('off')

                if contour is not None:
                    plt.subplot(rows, img_each_col * cols, img_each_col * k + 3)
                    plt.imshow(contour.cpu().numpy(), cmap='gist_gray')
                    plt.axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            plt.close()
            figure_list.append(buf.getvalue())

    return figure_list


def tensorboard_to_excel(log_path, save_log_name, save_log_path):
    merged_data = read_all_tensorboard_event_files(log_path)
    write_to_excel(merged_data, os.path.join(save_log_path, save_log_name))
    return f"Data from all TensorBoard event files in {log_path} has been merged and written to {save_log_name}."


def get_stats_between_maps(stat_mode, predicted_path, groundtruth_path, iou_threshold):
    ground_truth = Utils.path_to_array(groundtruth_path, label=True)
    predicted = Utils.path_to_array(predicted_path, label=True)
    if stat_mode == 'Semantic':
        ground_truth = torch.clamp(torch.from_numpy(ground_truth), 0, 1)
        predicted = torch.clamp(torch.from_numpy(predicted), 0, 1)
        intersection = 2 * torch.sum(ground_truth * predicted) + 0.001
        union = torch.sum(predicted) + torch.sum(ground_truth) + 0.001
        tp = (predicted * ground_truth).sum()
        fn = ((1 - predicted) * ground_truth).sum()
        tn = ((1 - predicted) * (1 - ground_truth)).sum()
        fp = (predicted * (1 - ground_truth)).sum()
        dice = intersection / union
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        return [dice.item(), sensitivity.item(), specificity.item(),
                fpr.item(), fnr.item()]
    else:
        tpr, fpr, fnr, precision, recall = Metrics.instance_segmentation_metrics(
            predicted, ground_truth, iou_threshold)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return [tpr.item(), fpr.item(), fnr.item(), precision.item(), recall.item()]


def auto_precision_logic():
    if not torch.cuda.is_available():
        return '32', ('Warning: No GPU detected! Will default to fp32 and '
                      'training will be SLOW (if it happens at all)!'), 'warning'

    devices = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        gpu_type = classify_gpu_type(name)
        devices.append((name, gpu_type, i))
    devices.sort(key=lambda x: (0 if x[1] == 'discrete' else 1, x[2]))
    main_name, _, main_index = devices[0]

    if any(kw in main_name.lower() for kw in ['nvidia', 'geforce', 'quadro', 'tesla', 'titan', 'rtx']):
        capability = torch.cuda.get_device_capability(main_index)
        cc = capability[0] + capability[1] / 10.0
        if cc >= 8.0:
            return 'bf16-mixed', f'Newer Nvidia GPU detected: {main_name}!', 'info'
        elif cc >= 7.0:
            return '16-mixed', f'Older Nvidia GPU detected: {main_name}!', 'info'
        else:
            return '32', f'Really old Nvidia GPU detected: {main_name}, defaulting to fp32.', 'warning'

    if 'amd' in main_name.lower() or 'radeon' in main_name.lower():
        name_lower = main_name.lower()
        newer_amd_keywords = [
            'radeon rx 9', 'radeon rx 7', 'rdna 3', 'rdna 4',
            'radeon pro w7', 'radeon pro w9', 'ai pro r9',
            '780m', '760m', '740m', '890m', '880m', '8060s', '8050s',
            'mi200', 'mi210', 'mi250', 'mi300', 'mi325', 'mi350', 'mi355', 'mi400',
        ]
        if any(kw in name_lower for kw in newer_amd_keywords):
            return 'bf16-mixed', f'Newer AMD GPU detected: {main_name}!', 'info'
        elif 'mi100' in name_lower:
            return '16-mixed', 'MI100 GPU detected!', 'info'
        else:
            return '32', (f'Really old AMD GPU detected: {main_name}, defaulting to fp32. '
                          f'If you are using a newer GPU (2026+) and this shows up, '
                          f'please report it in the GitHub repository.'), 'warning'

    return '32', 'Unknown GPU type detected, defaulting to fp32.', 'warning'


# ---------------------------------------------------------------------------
# Image embedding (base64) for the accordion comparison figures
# ---------------------------------------------------------------------------

def _img_data_uri(rel_path, max_width=400, max_bytes=150_000):
    """Embed a reference image as a small data URI.

    WebView2's NavigateToString() rejects oversized HTML, so we downscale
    and re-encode before inlining. The CSS caps displayed width at 400 px,
    so this is lossless from the user's perspective.
    """
    try:
        with Image.open(rel_path) as img:
            img = img.convert('RGB')
            width = min(max_width, img.width)
            while width >= 80:
                if img.width != width:
                    ratio = width / img.width
                    resized = img.resize(
                        (width, max(1, int(img.height * ratio))),
                        Image.LANCZOS,
                    )
                else:
                    resized = img
                buf = io.BytesIO()
                resized.save(buf, format='JPEG', quality=80, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    b64 = base64.b64encode(data).decode('ascii')
                    return 'data:image/jpeg;base64,' + b64
                width = int(width * 0.7)
        print(f'Warning: could not shrink {rel_path} below {max_bytes} bytes; skipping.')
        return ''
    except Exception as e:
        print(f'Warning: could not embed {rel_path} as data URI: {e}')
        return ''


IMG_TOKENS = {
    '__IMG_ROI_TOO_SMALL__': 'GitHub_Res/roi too small.png',
    '__IMG_ROI_JUST_RIGHT__': 'GitHub_Res/roi just right.png',
    '__IMG_ROI_TOO_BIG__': 'GitHub_Res/roi too big.png',
    '__IMG_REQUIRE_BS2__': 'GitHub_Res/require bs 2.png',
    '__IMG_DO_NOT_REQUIRE_BS2__': 'GitHub_Res/do not require bs 2.png',
}


def build_html():
    html = HTML
    for token, path in IMG_TOKENS.items():
        html = html.replace(token, _img_data_uri(path))
    return html


# ---------------------------------------------------------------------------
# pywebview API
# ---------------------------------------------------------------------------

class WebUIApi:
    def __init__(self):
        self.executor = CommandExecutor(on_output=self._on_output,
                                        on_finish=self._on_finish)

    @staticmethod
    def _get_window():
        return webview.windows[0] if webview.windows else None

    def _on_output(self, line, mode='newline'):
        window = self._get_window()
        if window is None:
            return
        try:
            window.evaluate_js(f'appendLog({json.dumps(line)}, {json.dumps(mode)})')
        except Exception:
            pass

    def _on_finish(self, return_code):
        window = self._get_window()
        if window is None:
            return
        try:
            window.evaluate_js(f'onWorkflowFinished({json.dumps(return_code)})')
        except Exception:
            pass

    def open_folder(self):
        return open_folder_dialog()

    def open_file(self):
        return open_file_dialog()

    def count_num_image_files_api(self, path, exclude_labels=True):
        try:
            return {'status': 'ok', 'count': count_num_image_files(path, exclude_labels)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_auto_parameters_api(self, workflow_box, train_dataset_path, val_dataset_path,
                                hw_size, d_size, batch_size, train_steps,
                                enable_unsupervised, unsupervised_train_dataset_path):
        try:
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
            unsupervised_train_multiplier = (train_multiplier * num_t_files) // max(num_u_files, 1)
            num_epochs = steps_to_epochs(train_steps, train_multiplier, num_t_files) * batch_size

            return {
                'status': 'ok',
                'val_num_patch': val_num_patch,
                'num_t_files': num_t_files,
                'num_u_files': num_u_files,
                'train_multiplier': train_multiplier,
                'unsupervised_train_multiplier': unsupervised_train_multiplier,
                'num_epochs': num_epochs,
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def visualise_augmentations_api(self, train_dataset_path, hw_size, d_size,
                                    augmentation_csv, slice_to_show, num_copies,
                                    pairing_samples, segmentation_mode,
                                    contour_map_width, train_key_name):
        try:
            pngs = visualise_augmentations(
                train_dataset_path, hw_size, d_size, augmentation_csv,
                int(slice_to_show), int(num_copies), pairing_samples,
                segmentation_mode, int(contour_map_width), train_key_name)
            imgs = ['data:image/png;base64,' + base64.b64encode(p).decode() for p in pngs]
            return {'status': 'ok', 'images': imgs}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def tensorboard_to_excel_api(self, log_path, save_log_name, save_log_path):
        try:
            msg = tensorboard_to_excel(log_path, save_log_name, save_log_path)
            return {'status': 'ok', 'message': msg}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_stats_between_maps_api(self, stat_mode, predicted_path, groundtruth_path, iou_threshold):
        try:
            vals = get_stats_between_maps(stat_mode, predicted_path, groundtruth_path, float(iou_threshold))
            return {'status': 'ok', 'values': vals}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def auto_precision_api(self):
        try:
            precision, message, level = auto_precision_logic()
            return {'status': 'ok', 'precision': precision, 'message': message, 'level': level}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def is_workflow_running(self):
        return self.executor.is_running()

    def interrupt_workflow(self):
        return {'status': 'ok', 'killed': self.executor.kill_command()}

    def start_workflow(self, inputs):
        if self.executor.is_running():
            return {'success': False, 'message': 'A workflow is already running.'}
        try:
            cmd = self._build_command(inputs)
        except Exception as e:
            return {'success': False, 'message': f'Failed to build command: {e}'}

        ok = self.executor.execute_command(cmd)
        if not ok:
            return {'success': False, 'message': 'Failed to start subprocess.'}
        return {'success': True, 'command': cmd}

    @staticmethod
    def _build_command(inputs):
        wf_str = " ".join(inputs['workflow_box'])
        existing_model_arg = ''
        if inputs.get('existing_model_path'):
            existing_model_arg = f'--existing_model_path "{inputs["existing_model_path"]}"'

        parts = [
            f'"{sys.executable}"', 'workflow.py',
            '--workflow_box', wf_str,
            '--segmentation_mode', inputs['segmentation_mode'],
            '--train_dataset_path', inputs['train_dataset_path'],
            '--augmentation_csv_path', f'"{inputs["augmentation_csv_path"]}"',
            '--train_multiplier', str(inputs['train_multiplier']),
            '--batch_size', str(inputs['batch_size']),
            '--num_epochs', str(inputs['num_epochs']),
            '--unsupervised_train_dataset_path', inputs['unsupervised_train_dataset_path'],
            '--unsupervised_train_multiplier', str(inputs['unsupervised_train_multiplier']),
            '--tensorboard_path', inputs['tensorboard_path'],
            '--val_dataset_path', inputs['val_dataset_path'],
            '--test_dataset_path', inputs['test_dataset_path'],
            '--predict_dataset_path', inputs['predict_dataset_path'],
            '--precision', inputs['precision'],
            '--save_model_name', inputs['save_model_name'],
            '--save_model_path', inputs['save_model_path'],
            '--train_key_name', inputs['train_key_name'],
            '--val_key_name', inputs['val_key_name'],
            '--test_key_name', inputs['test_key_name'],
            '--predict_key_name', inputs['predict_key_name'],
            '--hw_size', str(inputs['hw_size']),
            '--d_size', str(inputs['d_size']),
            '--predict_hw_size', str(inputs['predict_hw_size']),
            '--predict_depth_size', str(inputs['predict_depth_size']),
            '--predict_hw_overlap', str(inputs['predict_hw_overlap']),
            '--predict_depth_overlap', str(inputs['predict_depth_overlap']),
            '--watershed_dynamic', str(inputs['watershed_dynamic']),
            '--result_folder_path', inputs['result_folder_path'],
            '--model_architecture', inputs['model_architecture'],
            '--model_patch_size_xy', str(inputs['model_patch_size_xy']),
            '--model_patch_size_z', str(inputs['model_patch_size_z']),
            '--model_depth', str(inputs['model_depth']),
            '--z_to_xy_ratio', str(inputs['z_to_xy_ratio']),
            '--model_depth_multiplier', str(inputs['model_depth_multiplier']),
            '--train_dataset_mode', f'"{inputs["train_dataset_mode"]}"',
            '--contour_map_width', str(inputs['contour_map_width']),
            '--val_dataset_mode', f'"{inputs["val_dataset_mode"]}"',
            '--test_dataset_mode', f'"{inputs["test_dataset_mode"]}"',
        ]

        if existing_model_arg:
            parts.append(existing_model_arg)
        if inputs['pairing_samples']:
            parts.append('--pairing_samples')
        if inputs['enable_tensorboard']:
            parts.append('--enable_tensorboard')
        if inputs['enable_unsupervised']:
            parts.append('--enable_unsupervised')
        if inputs['memory_saving_mode']:
            parts.append('--memory_saving_mode')
        if inputs['read_existing_model']:
            parts.append('--read_existing_model')
        if not inputs['enable_mid_visualization']:
            parts.append('--mid_visualization')
        if inputs['train_offload']:
            parts.append('--train_offload')
        if inputs['val_offload']:
            parts.append('--val_offload')
        if inputs['test_offload']:
            parts.append('--test_offload')
        if inputs['predict_offload']:
            parts.append('--predict_offload')
        if inputs['instance_seg_mode']:
            parts.append('--instance_seg_mode')
        if inputs['pixel_reclaim']:
            parts.append('--pixel_reclaim')

        return ' '.join(parts)


# ---------------------------------------------------------------------------
# HTML / JS / CSS
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Volume Seg Tool</title>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f5f5f5; color: #222; }
.tabs { display: flex; background: #2c3e50; }
.tab-btn { padding: 10px 22px; background: transparent; color: #ecf0f1; border: none; cursor: pointer; font-size: 14px; }
.tab-btn.active { background: #34495e; }
.tab-btn:hover { background: #3b5998; }
.tab-content { display: none; padding: 15px; }
.tab-content.active { display: block; }
section { border: 1px solid #d0d0d0; padding: 12px 14px; margin-bottom: 12px; border-radius: 4px; background: #fff; }
section h3 { margin-top: 0; margin-bottom: 8px; font-size: 15px; color: #2c3e50; }
label { display: inline-block; margin: 4px 12px 4px 0; font-size: 13px; }
/* Native tooltip cursor — labels with a `title=` show the help cursor */
label[title] { cursor: help; }
input[type=text], input[type=number], select { padding: 4px 6px; border: 1px solid #ccc; border-radius: 3px; font-size: 13px; width: 220px; margin-left: 4px; }
input[type=text]:read-only, input[type=number]:read-only { background: #eee; color: #666; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 4px; }
.btn { padding: 7px 14px; cursor: pointer; border: 1px solid transparent; border-radius: 4px; font-size: 13px; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-primary { background: #2980b9; color: white; }
.btn-primary:hover:not(:disabled) { background: #2471a3; }
.btn-danger { background: #c0392b; color: white; }
.btn-danger:hover:not(:disabled) { background: #a93226; }
.btn-default { background: #ecf0f1; border-color: #bdc3c7; }
.btn-default:hover:not(:disabled) { background: #dfe4e5; }
.btn-icon { padding: 4px 8px; }
#log-output { background: #1e1e1e; color: #d4d4d4; padding: 10px; height: 220px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; border-radius: 4px; }
#log-output div { padding: 1px 0; }
#status-bar { padding: 10px 15px; background: #ecf0f1; font-weight: bold; font-size: 13px; border-bottom: 1px solid #ccc; }
.status-idle { background: #ecf0f1 !important; color: #555; }
.status-running { background: #f39c12 !important; color: #fff; }
.status-finished { background: #27ae60 !important; color: #fff; }
.status-failed { background: #c0392b !important; color: #fff; }
.gallery { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
.gallery img { max-width: 500px; border: 1px solid #ccc; border-radius: 3px; }
.toast { position: fixed; top: 20px; right: 20px; padding: 12px 18px; border-radius: 4px; z-index: 1000;
         box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 13px; max-width: 400px; }
.toast.info { background: #d1ecf1; color: #0c5460; }
.toast.warning { background: #fff3cd; color: #856404; }
.toast.error { background: #f8d7da; color: #721c24; }
.hidden { display: none !important; }
.hint { font-size: 12px; color: #666; margin: 4px 0; }
.note { font-size: 12px; color: #6c4400; background: #fff8e1; padding: 6px 10px;
        border-left: 3px solid #f39c12; border-radius: 3px; margin: 6px 0; }

/* Accordions (using <details>) */
details.accordion { border: 1px solid #d0d0d0; border-radius: 4px; margin: 8px 0; background: #fafafa; }
details.accordion > summary { cursor: pointer; padding: 8px 12px; font-weight: bold; font-size: 14px; color: #2c3e50; user-select: none; }
details.accordion > summary:hover { background: #eef0f1; }
details.accordion[open] > summary { border-bottom: 1px solid #d0d0d0; }
.accordion-body { padding: 10px 14px; font-size: 13px; }
.accordion-body p { margin: 4px 0; }
.accordion-body .img-row { display: flex; gap: 12px; flex-wrap: wrap; }
.accordion-body figure { margin: 0; flex: 1; min-width: 220px; max-width: 400px; }
.accordion-body figure img { width: 100%; border: 1px solid #ccc; border-radius: 3px; background: #fff; min-height: 60px; }
.accordion-body figcaption { font-size: 12px; color: #555; margin-top: 4px; text-align: center; }
.workflow-checkboxes label { margin-right: 16px; }
</style>
</head>
<body>

<div id="status-bar" class="status-idle">Idle</div>

<div class="tabs">
  <button class="tab-btn active" data-tab="main">Main</button>
  <button class="tab-btn" data-tab="extras">Extras</button>
</div>

<!-- ==================== MAIN TAB ==================== -->
<div id="main-tab" class="tab-content active">

  <section>
    <h3>Workflow</h3>
    <div class="row">
      <span>Segmentation Mode:</span>
      <label><input type="radio" name="segmentation_mode" value="Semantic" checked onchange="segmentationModeChanged()"> Semantic</label>
      <label><input type="radio" name="segmentation_mode" value="Instance" onchange="segmentationModeChanged()"> Instance</label>
    </div>
    <div class="row workflow-checkboxes">
      <span>Components to enable:</span>
      <label><input type="checkbox" id="wf_training" checked> Training</label>
      <label><input type="checkbox" id="wf_validation" checked> Validation</label>
      <label><input type="checkbox" id="wf_test"> Test</label>
      <label><input type="checkbox" id="wf_predict" checked> Predict</label>
    </div>
    <details class="accordion">
      <summary>Workflows Explanation</summary>
      <div class="accordion-body">
        <p>Training: Train a network using the training dataset.</p>
        <p>Validation: Use a Validation Dataset to evaluate the segmentation quality during training.</p>
        <p>Test: Similar to Validation, but only run once after the model has finished training.</p>
        <p>Predict: Use a (trained) network to predict the label of the predict dataset.</p>
        <p>Note that if the validation is not enabled, the script will use train dataset loss to determine when to reduce the learning rate.</p>
      </div>
    </details>
  </section>

  <section>
    <h3>Dataset Information</h3>
    <p class="hint">Necessary questions regarding the dataset in order to compute some hyperparameters.</p>
    <p class="hint">Should always be filled.</p>
    <div class="row">
      <label title="Given a square shaped 2d patch randomly taken from your dataset, what's the minimum side length (in pixels) you (as a human) would need to tell and segment the cellular structure of interest from the patch? Is used to compute the size of images send into the network.">Size to spot feature: <input type="number" id="size_to_spot" value="32" step="1" oninput="calculateDHWSize()"></label>
      <label title="The ratio of the z-resolution of the images in the dataset to their xy-resolution. Usually is 1.0 or larger if the dataset is anisotropic (e.g. SBF-SEM). We assume xy has the same resolution.">Z-resolution to XY-resolution ratio: <input type="number" id="z_to_xy_ratio" value="1.0" step="0.1" oninput="calculateDHWSize()"></label>
      <label>Check to allow manual editing of the parameters below, instead of being calculated automatically <input type="checkbox" id="manual_mode" onchange="manualModeChanged()"></label>
    </div>
    <details class="accordion">
      <summary>Example of various field of view</summary>
      <div class="accordion-body">
        <div class="img-row">
          <figure>
            <img src="__IMG_ROI_TOO_SMALL__" alt="Field of view too small" onerror="this.alt='(GitHub_Res/roi too small.png not found)'">
            <figcaption>Field of view too small, cannot tell the feature</figcaption>
          </figure>
          <figure>
            <img src="__IMG_ROI_JUST_RIGHT__" alt="Field of view just right" onerror="this.alt='(GitHub_Res/roi just right.png not found)'">
            <figcaption>Field of view just right</figcaption>
          </figure>
          <figure>
            <img src="__IMG_ROI_TOO_BIG__" alt="Field of view too big" onerror="this.alt='(GitHub_Res/roi too big.png not found)'">
            <figcaption>Field of view too big, unnecessary computation and memory use</figcaption>
          </figure>
        </div>
      </div>
    </details>
    <details class="accordion">
      <summary>Automatically computed hyperparameters</summary>
      <div class="accordion-body">
        <div class="row">
          <label>Height and Width (px) of training image: <input type="number" id="hw_size" value="64" readonly oninput="calculatePredictParameters()"></label>
          <label>Depth (px) of training image: <input type="number" id="d_size" value="64" readonly oninput="calculatePredictParameters()"></label>
        </div>
        <div class="row">
          <label>Patch Height and Width (px): <input type="number" id="model_patch_size_xy" value="4" readonly oninput="calculatePredictParameters()"></label>
          <label>Patch Depth (px): <input type="number" id="model_patch_size_z" value="4" readonly oninput="calculatePredictParameters()"></label>
          <label>Number of Transformer layers: <input type="number" id="model_depth" value="8" readonly></label>
        </div>
      </div>
    </details>
    <div class="row" style="margin-top:8px">
      <label title="Allow using unlabelled data to enhance performance."><input type="checkbox" id="enable_unsupervised" onchange="enableUnsupervisedChanged()"> Enable Unsupervised Learning</label>
    </div>
    <div id="unsupervised_section" class="hidden">
      <div class="row">
        <label>Unsupervised Train Dataset Path:
          <input type="text" id="unsupervised_train_dataset_path" value="Datasets/unsupervised_train">
        </label>
        <button class="btn btn-default btn-icon" onclick="pickFolder('unsupervised_train_dataset_path')">📄</button>
      </div>
      <div class="row">
        <label>Number of image files in the unsupervised training set: <input type="number" id="num_u_files" value="1" readonly></label>
        <label>Unsupervised Train Multiplier (Repeats): <input type="number" id="unsupervised_train_multiplier" value="1" readonly></label>
      </div>
    </div>
  </section>

  <section>
    <h3>Validation Settings</h3>
    <div class="row">
      <label title="If you are using hdf5 type image, you need to provide its data set name so VST could extract the correct data from it.">hdf5 dataset name: <input type="text" id="val_key_name" value="Default"></label>
    </div>
    <div class="row">
      <label title="Instead of having all the validation data in memory, offload them to your disk and load only the parts in need. Can reduce memory use. At the expense of your disk space and potentially slower training. Allows datasets that are larger than your memory. Each individual file still needs to be smaller, though."><input type="checkbox" id="val_offload"> Offload validation dataset to disk</label>
    </div>
    <div class="row">
      <label>Validation Dataset Path: <input type="text" id="val_dataset_path" value="Datasets/val"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('val_dataset_path')">📄</button>
    </div>
    <div class="row">
      <label>Number of patches in validation set, automatically computed: <input type="number" id="val_num_patch" readonly></label>
      <label>Dataset Mode:
        <select id="val_dataset_mode">
          <option>Fully Labelled</option>
          <option>Sparsely Labelled</option>
        </select>
      </label>
    </div>
  </section>

  <section>
    <h3>Training Settings</h3>
    <div class="row">
      <label title="If you are using hdf5 type image, you need to provide its dataset name.">hdf5 dataset name: <input type="text" id="train_key_name" value="Default"></label>
    </div>
    <div class="row">
      <label title="Instead of having all the training data (including unsupervised) in memory, offload them to your disk and load only the parts in need. Can reduce memory use. At the expense of your disk space and potentially slower training. Allows datasets that are larger than your memory. Each individual file still needs to be smaller, though."><input type="checkbox" id="train_offload"> Offload training dataset to disk</label>
    </div>
    <div class="row">
      <label>Train Dataset Path: <input type="text" id="train_dataset_path" value="Datasets/train"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('train_dataset_path')">📄</button>
    </div>
    <div class="row">
      <label title="Number of training patch to feed into the network at once.">Batch Size: <input type="number" id="batch_size" value="1" readonly></label>
      <label title="Tick this if your training data contain large area without any foreground object, or large area without any background object. If in doubt, tick it."><input type="checkbox" id="pairing_samples" onchange="pairingChanged()"> Pairing positive and negative samples</label>
      <label>Number of image files in the training set: <input type="number" id="num_t_files" readonly></label>
    </div>
    <details class="accordion">
      <summary>When do you need to pair positive and negative sample</summary>
      <div class="accordion-body">
        <div class="img-row">
          <figure>
            <img src="__IMG_REQUIRE_BS2__" alt="Requires pairing" onerror="this.alt='(GitHub_Res/require bs 2.png not found)'">
            <figcaption>Require the box to be ticked due to large area of empty space outside the cell</figcaption>
          </figure>
          <figure>
            <img src="__IMG_DO_NOT_REQUIRE_BS2__" alt="Does not require pairing" onerror="this.alt='(GitHub_Res/do not require bs 2.png not found)'">
            <figcaption>Does not Require the box to be ticked</figcaption>
          </figure>
        </div>
      </div>
    </details>
    <div class="row">
      <label title="Automatically calculated to aim for 10% steps in validation, 90% in train. Limit to a max of 128.">Train Multiplier (Repeats): <input type="number" id="train_multiplier" readonly></label>
    </div>
    <div class="row">
      <label>Training Duration:
        <select id="train_length" onchange="trainLengthChanged()">
          <option>Short</option>
          <option>Medium</option>
          <option>Long</option>
          <option>Custom</option>
        </select>
      </label>
      <label>Training Steps: <input type="number" id="train_steps" value="2000" readonly></label>
      <label>Maximum Number of Epochs, automatically calculated: <input type="number" id="num_epochs" readonly></label>
    </div>
    <div class="row">
      <button class="btn btn-default" onclick="calculateRepeats()">Calculate Training Repeats and Epochs</button>
    </div>
    <!-- enable_tensorboard was `visible=False` in the original — hidden here too, but defaults to True -->
    <input type="checkbox" id="enable_tensorboard" checked class="hidden">
    <div class="row">
      <label>Path to the folder which the training log will be save to: <input type="text" id="tensorboard_path" value="lightning_logs"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('tensorboard_path')">📄</button>
    </div>
    <div class="row">
      <label>Dataset Mode:
        <select id="train_dataset_mode">
          <option>Fully Labelled</option>
          <option>Sparsely Labelled</option>
        </select>
      </label>
      <label id="contour_map_width_row" class="hidden">Width of contour (outward): <input type="number" id="contour_map_width" value="1" min="1"></label>
    </div>
    <div class="row">
      <label>Csv File for Data Augmentation Settings. You should change it to the isotropic version if your data have isotropic resolution.: <input type="text" id="augmentation_csv_path" value="Augmentation Parameters Anisotropic.csv" style="width:340px"></label>
      <button class="btn btn-default btn-icon" onclick="pickFile('augmentation_csv_path')">📄</button>
    </div>
    <div class="row">
      <button class="btn btn-primary" onclick="showAugmentations(this)">Show some example patches from your training dataset, under the current patch size and augmentation settings. (Warning: fairly slow, and do not click if your training dataset is larger than your RAM)</button>
    </div>
    <div id="aug_gallery" class="gallery"></div>
  </section>

  <section>
    <h3>Test Settings</h3>
    <p class="note">Due to the limitation of the built-in test function. The dice score obtained via the test workflow may not represent the actual overall dice score.</p>
    <p class="note">If you want to obtain the actual dice score, it's recommended to process the image via the Predict workflow first. Then use the 'Extras' tab above to extract the overall dice score.</p>
    <div class="row">
      <label title="If you are using hdf5 type image, you need to provide its dataset name.">hdf5 dataset name: <input type="text" id="test_key_name" value="Default"></label>
    </div>
    <div class="row">
      <label title="Instead of having all the test data in memory, offload them to your disk and load only the parts in need. Can reduce memory use. At the expense of your disk space and slower starting up time. Allows datasets that are larger than your memory. Each individual file still needs to be smaller, though."><input type="checkbox" id="test_offload"> Offload test dataset to disk</label>
    </div>
    <div class="row">
      <label>Test Dataset Path: <input type="text" id="test_dataset_path" value="Datasets/test"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('test_dataset_path')">📄</button>
    </div>
    <div class="row">
      <label>Dataset Mode:
        <select id="test_dataset_mode">
          <option>Fully Labelled</option>
          <option>Sparsely Labelled</option>
        </select>
      </label>
    </div>
  </section>

  <section>
    <h3>Predict Settings</h3>
    <div class="row">
      <label title="If you are using hdf5 type image, you need to provide its dataset name.">hdf5 dataset name: <input type="text" id="predict_key_name" value="Default"></label>
    </div>
    <div class="row">
      <label title="Instead of having all the predict data in memory, offload them to your disk and load only the parts in need. Can reduce memory use. At the expense of your disk space and potentially slower predicting. Allows datasets that are larger than your memory. Each individual file still needs to be smaller, though."><input type="checkbox" id="predict_offload"> Offload predict dataset to disk</label>
    </div>
    <div class="row">
      <label>Predict Dataset Path: <input type="text" id="predict_dataset_path" value="Datasets/predict"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('predict_dataset_path')">📄</button>
    </div>
    <div class="row">
      <label>Expansion in Height and Width for each Patch (px): <input type="number" id="predict_hw_overlap" value="8" readonly></label>
      <label>Expansion in Depth for each Patch (px): <input type="number" id="predict_depth_overlap" value="8" readonly></label>
      <label>Height and Width of each Patch (px): <input type="number" id="predict_hw_size" value="48" readonly></label>
      <label>Depth of each Patch (px): <input type="number" id="predict_depth_size" value="48" readonly></label>
    </div>
    <div class="row">
      <label>Result Folder Path: <input type="text" id="result_folder_path" value="Datasets/result"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('result_folder_path')">📄</button>
    </div>
    <div class="row">
      <label title="Use a slower and more memory intensive watershed method for separate touching objects, instead of simple connected component labelling. Will yield result with much less under-segment objects."><input type="checkbox" id="instance_seg_mode"> Use distance transform watershed for instance segmentation</label>
      <label>Dynamic of intensity for the search of regional minima in the distance transform image. Increasing its value will yield more object merges.: <input type="number" id="watershed_dynamic" value="10"></label>
    </div>
    <div class="row">
      <label title="Due to how instance segmentation works, some pixels will be lost when separating touching objects, this settings will try to reclaim some of those lost pixels, but can take quite some time."><input type="checkbox" id="pixel_reclaim"> Enable Pixel reclaim operation for instance segmentation</label>
    </div>
  </section>

  <section>
    <h3>Model & Training Controls</h3>
    <div class="row">
      <label title="Else it will create a new model with randomised weight."><input type="checkbox" id="read_existing_model" onchange="readExistingModelChanged()"> Read Existing Model Weight File</label>
    </div>
    <div class="row">
      <label>Path to Existing Model Weight File, it should be a file with 'ckpt' at the end.: <input type="text" id="existing_model_path" value="trained_model/example_name.ckpt"></label>
      <button class="btn btn-default btn-icon" onclick="pickFile('existing_model_path')">📄</button>
    </div>
    <div id="model_config_section">
      <div class="row">
        <label>Model Architecture:
          <select id="model_architecture">
            <option>SwinTransformer</option>
          </select>
        </label>
        <label title="Increasing this numbe makes the model deeper, which can help if you want more accuracy at the expense of speed and video memory usage.">Model Depth multiplier: <input type="number" id="model_depth_multiplier" value="1.0" step="0.1"></label>
      </div>
    </div>
    <!-- enable_mid_visualization lived inside a `visible=False` accordion in the original.
         Hidden here too, but defaults to True. -->
    <input type="checkbox" id="enable_mid_visualization" checked class="hidden">
    <div class="row">
      <label title="If you are experiencing running out of system memory (Not CUDA memory!) during training, This option could help by using only 1 thread to do data loading. Can significantly slow down the training if your system have low single core performance."><input type="checkbox" id="memory_saving_mode"> Memory Saving Mode</label>
    </div>
    <div class="row">
      <label title="fp16 or bf16 precision could significantly cut the VRAM usage.bf16 is recommended over fp16 if you are using a newer GPU.">Precision:
        <select id="precision">
          <option value="32">32</option>
          <option value="16-mixed">16-mixed</option>
          <option value="bf16-mixed">bf16-mixed</option>
        </select>
      </label>
      <button class="btn btn-default" onclick="autoPrecision()">Find suitable precision based on your GPU</button>
    </div>
    <div class="row">
      <label>File Name for Model Saved, do not include extension: <input type="text" id="save_model_name" value="example_name"></label>
      <label title="For path with space in it, put '' on both sides">Path to Save the Model Weight: <input type="text" id="save_model_path" value="trained_model"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('save_model_path')">📄</button>
    </div>
  </section>

  <div class="row" style="margin-bottom:12px">
    <button id="start-btn" class="btn btn-primary" onclick="startWorkflow()" style="font-size:15px;padding:10px 24px">Start!</button>
    <button id="interrupt-btn" class="btn btn-danger" onclick="interruptWorkflow()" disabled style="font-size:15px;padding:10px 24px">Interrupt!</button>
  </div>

  <section>
    <h3>Live Output</h3>
    <div id="log-output"></div>
  </section>
</div>

<!-- ==================== EXTRAS TAB ==================== -->
<div id="extras-tab" class="tab-content">

  <section>
    <h3>Output TensorBoard log to Excel</h3>
    <div class="row">
      <label>Path to the folder which the tensorboard log are saved to: <input type="text" id="tensorboard_path_e" value="lightning_logs"></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('tensorboard_path_e')">📄</button>
    </div>
    <div class="row">
      <label>File name for the output Excel file, including extension: <input type="text" id="save_log_name" value="output.xlsx"></label>
      <label>Path to Save the Excel file: <input type="text" id="save_log_path" value=""></label>
      <button class="btn btn-default btn-icon" onclick="pickFolder('save_log_path')">📄</button>
    </div>
    <div class="row">
      <button class="btn btn-primary" onclick="tensorboardToExcel()">Output TensorBoard log to Excel</button>
    </div>
  </section>

  <section>
    <h3>Calculate statistics between the predicted labels and ground truth labels</h3>
    <p class="note">Calculate instance segmentation metrics is hard and VST uses your GPU to accelerate the process. It would be painfully slow if you are using a CPU! And still not fast if you do have a GPU...</p>
    <div class="row">
      <span>Mode:</span>
      <label><input type="radio" name="stat_mode" value="Semantic" checked onchange="statModeChanged()"> Semantic</label>
      <label><input type="radio" name="stat_mode" value="Instance" onchange="statModeChanged()"> Instance</label>
    </div>
    <div class="row">
      <label>Path to the predicted image: <input type="text" id="predicted_img_path" style="width:340px"></label>
      <button class="btn btn-default btn-icon" onclick="pickFile('predicted_img_path')">📄</button>
    </div>
    <div class="row">
      <label>Path to the ground truth image: <input type="text" id="ground_truth_img_path" style="width:340px"></label>
      <button class="btn btn-default btn-icon" onclick="pickFile('ground_truth_img_path')">📄</button>
    </div>
    <div class="row" id="iou_threshold_row" style="display:none">
      <label>IOU threshold: <input type="number" id="iou_threshold" value="0.5" step="0.05"></label>
    </div>
    <div class="row">
      <button class="btn btn-primary" onclick="getStats()">Get statistics!</button>
    </div>
    <div class="row">
      <label><span id="stat-label-0">Dice Score</span>: <input type="number" id="stat-output-0" readonly></label>
      <label><span id="stat-label-1">Sensitivity (True Positive Rate)</span>: <input type="number" id="stat-output-1" readonly></label>
      <label><span id="stat-label-2">Specificity (True Negative Rate)</span>: <input type="number" id="stat-output-2" readonly></label>
      <label><span id="stat-label-3">False Positive Rate</span>: <input type="number" id="stat-output-3" readonly></label>
      <label><span id="stat-label-4">False Negative Rate</span>: <input type="number" id="stat-output-4" readonly></label>
    </div>
  </section>
</div>

<script>
// ---------- utilities ----------
function $(id) { return document.getElementById(id); }
function numVal(id) { const v = parseFloat($(id).value); return isNaN(v) ? 0 : v; }
function intVal(id) { const v = parseInt($(id).value); return isNaN(v) ? 0 : v; }
function strVal(id) { return $(id).value; }
function boolVal(id) { return $(id).checked; }
function radioVal(name) { return document.querySelector(`input[name="${name}"]:checked`).value; }
const WF_IDS = { 'wf_training': 'Training', 'wf_validation': 'Validation', 'wf_test': 'Test', 'wf_predict': 'Predict' };
function checkedWorkflows() {
  return Object.keys(WF_IDS).filter(id => $(id).checked).map(id => WF_IDS[id]);
}

// ---------- tabs ----------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.tab === 'main' ? 'main-tab' : 'extras-tab';
    $(target).classList.add('active');
  });
});

// ---------- toast ----------
function showToast(message, level) {
  const toast = document.createElement('div');
  toast.className = 'toast ' + (level || 'info');
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}

// ---------- status bar ----------
function setStatus(text, cls) {
  const bar = $('status-bar');
  bar.textContent = text;
  bar.className = 'status-' + cls;
}

// ---------- logging ----------
window.appendLog = function(line, mode) {
  mode = mode || 'newline';
  const logEl = $('log-output');
  const wasAtBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;

  const replacePrevious =
    mode === 'carriage' &&
    logEl._lastWasCarriage &&
    logEl.lastElementChild &&
    line.length > 0;

  if (replacePrevious) {
    logEl.lastElementChild.textContent = line;
  } else {
    const lineEl = document.createElement('div');
    lineEl.textContent = line;
    logEl.appendChild(lineEl);
    while (logEl.children.length > 5000) logEl.removeChild(logEl.firstChild);
  }
  logEl._lastWasCarriage = (mode === 'carriage');
  if (wasAtBottom) logEl.scrollTop = logEl.scrollHeight;
};

// ---------- dataset info auto-calc ----------
function manualModeChanged() {
  const editable = $('manual_mode').checked;
  ['hw_size', 'd_size', 'model_patch_size_xy', 'model_patch_size_z', 'model_depth'].forEach(id => {
    $(id).readOnly = !editable;
  });
  if (!editable) calculateDHWSize();
}

function calculateDHWSize() {
  if ($('manual_mode').checked) return;
  const sizeToSpot = numVal('size_to_spot');
  const ratio = numVal('z_to_xy_ratio') || 1.0;

  const hwEstimate = 1.5 * sizeToSpot;
  const hwPatchSize = Math.trunc(Math.cbrt(64 * ratio));
  const dPatchSize = Math.trunc(Math.cbrt(64 / ratio));
  const windowSize = 4;
  const hwInterval = hwPatchSize * windowSize;
  const dInterval = dPatchSize * windowSize;
  const hwSize = hwInterval * Math.max(Math.round(hwEstimate / hwInterval), 1);
  const dSize = dInterval * Math.max(Math.round(hwEstimate / dInterval / ratio), 1);
  const depth = Math.ceil(hwEstimate / hwInterval);

  $('hw_size').value = hwSize;
  $('d_size').value = dSize;
  $('model_patch_size_xy').value = hwPatchSize;
  $('model_patch_size_z').value = dPatchSize;
  $('model_depth').value = depth;
  calculatePredictParameters();
}

function calculatePredictParameters() {
  const mpxy = intVal('model_patch_size_xy') || 4;
  const mpz = intVal('model_patch_size_z') || 4;
  const hwSize = intVal('hw_size') || 64;
  const dSize = intVal('d_size') || 64;

  const hwMult = hwSize <= 64 ? 1 : (hwSize <= 192 ? 2 : 3);
  const dMult = dSize <= 64 ? 1 : (dSize <= 192 ? 2 : 3);

  const hwOverlap = mpxy * 2 * hwMult;
  const dOverlap = mpz * 2 * dMult;
  $('predict_hw_overlap').value = hwOverlap;
  $('predict_depth_overlap').value = dOverlap;
  $('predict_hw_size').value = hwSize - 2 * hwOverlap;
  $('predict_depth_size').value = dSize - 2 * dOverlap;
}

// ---------- training settings ----------
function pairingChanged() {
  $('batch_size').value = $('pairing_samples').checked ? 2 : 1;
}

function trainLengthChanged() {
  const v = $('train_length').value;
  const steps = $('train_steps');
  if (v === 'Short') { steps.value = 2000; steps.readOnly = true; }
  else if (v === 'Medium') { steps.value = 5000; steps.readOnly = true; }
  else if (v === 'Long') { steps.value = 12000; steps.readOnly = true; }
  else { steps.readOnly = false; }
}

async function calculateRepeats() {
  const result = await pywebview.api.get_auto_parameters_api(
    checkedWorkflows(),
    strVal('train_dataset_path'),
    strVal('val_dataset_path'),
    intVal('hw_size'),
    intVal('d_size'),
    intVal('batch_size'),
    intVal('train_steps'),
    boolVal('enable_unsupervised'),
    strVal('unsupervised_train_dataset_path'),
  );
  if (result.status !== 'ok') {
    showToast('Error: ' + result.message, 'error');
    return;
  }
  $('val_num_patch').value = result.val_num_patch ?? '';
  $('num_t_files').value = result.num_t_files;
  $('num_u_files').value = result.num_u_files;
  $('train_multiplier').value = result.train_multiplier;
  $('unsupervised_train_multiplier').value = result.unsupervised_train_multiplier;
  $('num_epochs').value = result.num_epochs;
  showToast('Repeats and epochs calculated.', 'info');
}

// ---------- conditional visibility ----------
function segmentationModeChanged() {
  const mode = radioVal('segmentation_mode');
  $('contour_map_width_row').classList.toggle('hidden', mode !== 'Instance');
}

function enableUnsupervisedChanged() {
  $('unsupervised_section').classList.toggle('hidden', !$('enable_unsupervised').checked);
}

function readExistingModelChanged() {
  $('model_config_section').classList.toggle('hidden', $('read_existing_model').checked);
}

function statModeChanged() {
  const mode = radioVal('stat_mode');
  const labels = mode === 'Instance'
    ? ['True Positive Rate', 'False Positive Rate', 'False Negative Rate', 'Precision', 'Recall']
    : ['Dice Score', 'Sensitivity (True Positive Rate)', 'Specificity (True Negative Rate)',
       'False Positive Rate', 'False Negative Rate'];
  for (let i = 0; i < 5; i++) {
    $('stat-label-' + i).textContent = labels[i];
    $('stat-output-' + i).value = '';
  }
  $('iou_threshold_row').style.display = mode === 'Instance' ? 'flex' : 'none';
}

// ---------- augmentation gallery ----------
async function showAugmentations(btn) {
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Loading…';
  $('aug_gallery').innerHTML = '';
  try {
    const result = await pywebview.api.visualise_augmentations_api(
      strVal('train_dataset_path'),
      intVal('hw_size'),
      intVal('d_size'),
      strVal('augmentation_csv_path'),
      0, 6,
      boolVal('pairing_samples'),
      radioVal('segmentation_mode'),
      intVal('contour_map_width'),
      strVal('train_key_name'),
    );
    if (result.status !== 'ok') {
      showToast('Error: ' + result.message, 'error');
      return;
    }
    const gallery = $('aug_gallery');
    for (const src of result.images) {
      const img = document.createElement('img');
      img.src = src;
      gallery.appendChild(img);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

// ---------- extras ----------
async function tensorboardToExcel() {
  const result = await pywebview.api.tensorboard_to_excel_api(
    strVal('tensorboard_path_e'),
    strVal('save_log_name'),
    strVal('save_log_path'),
  );
  showToast(result.status === 'ok' ? result.message : ('Error: ' + result.message),
            result.status === 'ok' ? 'info' : 'error');
}

async function getStats() {
  const result = await pywebview.api.get_stats_between_maps_api(
    radioVal('stat_mode'),
    strVal('predicted_img_path'),
    strVal('ground_truth_img_path'),
    numVal('iou_threshold'),
  );
  if (result.status !== 'ok') {
    showToast('Error: ' + result.message, 'error');
    return;
  }
  for (let i = 0; i < 5; i++) $('stat-output-' + i).value = result.values[i];
}

// ---------- auto precision ----------
async function autoPrecision() {
  const result = await pywebview.api.auto_precision_api();
  if (result.status !== 'ok') {
    showToast('Error: ' + result.message, 'error');
    return;
  }
  $('precision').value = result.precision;
  if (result.message) showToast(result.message, result.level || 'info');
}

// ---------- workflow control ----------
function gatherInputs() {
  return {
    workflow_box: checkedWorkflows(),
    segmentation_mode: radioVal('segmentation_mode'),
    train_dataset_path: strVal('train_dataset_path'),
    augmentation_csv_path: strVal('augmentation_csv_path'),
    train_multiplier: intVal('train_multiplier'),
    batch_size: intVal('batch_size'),
    num_epochs: intVal('num_epochs'),
    enable_unsupervised: boolVal('enable_unsupervised'),
    unsupervised_train_dataset_path: strVal('unsupervised_train_dataset_path'),
    unsupervised_train_multiplier: intVal('unsupervised_train_multiplier'),
    enable_tensorboard: boolVal('enable_tensorboard'),
    tensorboard_path: strVal('tensorboard_path'),
    val_dataset_path: strVal('val_dataset_path'),
    test_dataset_path: strVal('test_dataset_path'),
    predict_dataset_path: strVal('predict_dataset_path'),
    read_existing_model: boolVal('read_existing_model'),
    existing_model_path: strVal('existing_model_path'),
    precision: strVal('precision'),
    save_model_name: strVal('save_model_name'),
    save_model_path: strVal('save_model_path'),
    train_key_name: strVal('train_key_name'),
    val_key_name: strVal('val_key_name'),
    test_key_name: strVal('test_key_name'),
    predict_key_name: strVal('predict_key_name'),
    train_offload: boolVal('train_offload'),
    val_offload: boolVal('val_offload'),
    test_offload: boolVal('test_offload'),
    predict_offload: boolVal('predict_offload'),
    hw_size: intVal('hw_size'),
    d_size: intVal('d_size'),
    predict_hw_size: intVal('predict_hw_size'),
    predict_depth_size: intVal('predict_depth_size'),
    predict_hw_overlap: intVal('predict_hw_overlap'),
    predict_depth_overlap: intVal('predict_depth_overlap'),
    watershed_dynamic: intVal('watershed_dynamic'),
    result_folder_path: strVal('result_folder_path'),
    enable_mid_visualization: boolVal('enable_mid_visualization'),
    model_architecture: strVal('model_architecture'),
    model_patch_size_xy: intVal('model_patch_size_xy'),
    model_patch_size_z: intVal('model_patch_size_z'),
    model_depth: intVal('model_depth'),
    z_to_xy_ratio: numVal('z_to_xy_ratio'),
    model_depth_multiplier: numVal('model_depth_multiplier'),
    train_dataset_mode: strVal('train_dataset_mode'),
    contour_map_width: intVal('contour_map_width'),
    val_dataset_mode: strVal('val_dataset_mode'),
    test_dataset_mode: strVal('test_dataset_mode'),
    pairing_samples: boolVal('pairing_samples'),
    memory_saving_mode: boolVal('memory_saving_mode'),
    instance_seg_mode: boolVal('instance_seg_mode'),
    pixel_reclaim: boolVal('pixel_reclaim'),
  };
}

function validateInputs(inputs) {
  if (inputs.workflow_box.length === 0) {
    showToast('Please select at least one workflow component.', 'warning');
    return false;
  }
  if (inputs.workflow_box.includes('Training') && !inputs.train_multiplier) {
    showToast('Please click "Calculate Training Repeats and Epochs" first!', 'warning');
    return false;
  }
  return true;
}

async function startWorkflow() {
  const inputs = gatherInputs();
  if (!validateInputs(inputs)) return;

  $('start-btn').disabled = true;
  $('interrupt-btn').disabled = false;
  setStatus('Running…', 'running');
  appendLog('===== Starting workflow =====');

  const result = await pywebview.api.start_workflow(inputs);
  if (!result.success) {
    $('start-btn').disabled = false;
    $('interrupt-btn').disabled = true;
    setStatus('Idle', 'idle');
    showToast('Failed to start workflow: ' + result.message, 'error');
    return;
  }
  appendLog('$ ' + result.command);
}

async function interruptWorkflow() {
  $('interrupt-btn').disabled = true;
  setStatus('Interrupting…', 'running');
  await pywebview.api.interrupt_workflow();
}

window.onWorkflowFinished = function(returnCode) {
  $('start-btn').disabled = false;
  $('interrupt-btn').disabled = true;
  if (returnCode === 0 || returnCode === null) {
    setStatus('Workflow finished.', 'finished');
    appendLog('===== Workflow finished =====');
  } else {
    setStatus('Workflow exited with code ' + returnCode + '.', 'failed');
    appendLog('===== Workflow exited with code ' + returnCode + ' =====');
  }
};

// initial UI state
calculatePredictParameters();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    api = WebUIApi()
    window = webview.create_window(
        'Volume Seg Tool',
        html=build_html(),
        width=1400,
        height=900,
        resizable=True,
    )

    # Expose individual callables to avoid pywebview serializing the whole
    # API object (which previously triggered the AccessibilityObject spam).
    for name in dir(api):
        if name.startswith('_'):
            continue
        attr = getattr(api, name)
        if callable(attr):
            window.expose(attr)

    webview.start(debug=False)


if __name__ == "__main__":
    main()