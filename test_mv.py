import os
import csv
import cv2
import yaml
import torch
import models
import copy
import pandas as pd
import datasets
import argparse
import numpy as np
import torch.nn as nn
import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import Levenshtein
from train import make_dataloader
from collections import defaultdict

from matplotlib import pyplot as plt
from utils_test import majority_vote_by_character, select_highest_confidence_string, select_most_frequent_string
import torchvision.transforms as T
import kornia as K
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
import torch.nn.functional as F
import random
import time
from pynvml import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resize_fn(img, size):
    return T.ToTensor()(
        T.Resize(size, T.InterpolationMode.BICUBIC)(T.ToPILImage()(img))
    )

def prepare_testing():
    test_loader = make_dataloader(config['test_dataset'], tag='test')

    sr_ckpt = None
    if config['model'] is not None:
        sr_cfg = config['model']
        sr_ckpt = torch.load(sr_cfg['load'])
        model_sr, _ = models.make(sr_ckpt['model'], load_model=True)
        model_sr = model_sr.cuda()
    else:
        model_sr = None

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and model_sr is not None:
        model_sr = nn.parallel.DataParallel(model_sr)

    if config['model_ocr']['name'] == 'ocr':
        model_ocr = models.make(config['model_ocr']).cuda()
    else:
        ocr_cfg = config['model_ocr']
        ocr_ckpt = torch.load(ocr_cfg['load'])
        model_ocr = models.make(ocr_ckpt['model'], load_model=True).cuda()

    return test_loader, model_sr, model_ocr

def process_results(preds, preds_conf_mean, t, mv_character_results, mv_hc_results, mv_results, res_type):
    # Append results to each dictionary for the specified resolution type and threshold
    mv_character_results[res_type][t].append(majority_vote_by_character(preds, t))
    mv_hc_results[res_type][t].append(select_highest_confidence_string(preds_conf_mean, preds, t))
    mv_results[res_type][t].append(select_most_frequent_string(preds_conf_mean, preds, t))

def calculate_levenshtein_batch(predictions, gt_batch):
    lev_distances = {}

    # Loop over each image in the batch
    for idx, gt in enumerate(gt_batch):
        lev_distances[idx] = {}  # Create a dictionary for each image

        # Loop over resolution types ('lr', 'hr', 'sr')
        for res_type, thresholds in predictions.items():
            lev_distances[idx][res_type] = {}

            # Loop over thresholds (1, 3, 5)
            for t, pred_list in thresholds.items():
                # Get the predicted strings for the current threshold
                pred_str = pred_list[idx]  # Use the prediction for the current image in the batch
                gt_str = gt  # Ground truth string (assuming it's a list with a single string)
                # Calculate the Levenshtein distance between predicted and ground truth
                lev_dist = Levenshtein.distance(pred_str, gt_str)

                lev_distances[idx][res_type][t] = 7-lev_dist

    return lev_distances

def count_distances_and_percentages_by_res_and_threshold(levenshtein_results, total_images, upper=1):
    # Initialize dictionaries to store the counts of distances for each resolution type and threshold
    distance_counts_by_res_and_threshold = {
        'lr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'hr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'sr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)}
    }
    
    # Initialize dictionaries to store the percentages of distances for each resolution type and threshold
    distance_percentages_by_res_and_threshold = {
        'lr': {1: {}, 3: {}, 5: {}},
        'hr': {1: {}, 3: {}, 5: {}},
        'sr': {1: {}, 3: {}, 5: {}}
    }

    # Iterate through each image's Levenshtein results
    for idx, res_data in levenshtein_results.items():
        # Iterate through each resolution type ('lr', 'hr', 'sr')
        for res_type, thresholds in res_data.items():
            # Iterate through each threshold (1, 3, 5)
            for t, dist in thresholds.items():
                # Increment the count for the corresponding distance for the specific res_type and threshold
                distance_counts_by_res_and_threshold[res_type][t][dist] += 1

    # Calculate the percentages for each resolution type and threshold
    for res_type in ['lr', 'hr', 'sr']:
        for t in [1, 3, 5]:
            for dist, count in distance_counts_by_res_and_threshold[res_type][t].items():
                distance_percentages_by_res_and_threshold[res_type][t][dist] = round((count / total_images) * 100, upper)

    return distance_counts_by_res_and_threshold, distance_percentages_by_res_and_threshold

def calculate_distances_and_percentages(mv_character_results, mv_hc_results, mv_results, gt):
    """Calculate Levenshtein results and distance percentages."""
    levenshtein_results_mvcp = calculate_levenshtein_batch(mv_character_results, gt)
    levenshtein_results_hc = calculate_levenshtein_batch(mv_hc_results, gt)
    levenshtein_results_mv = calculate_levenshtein_batch(mv_results, gt)

    _, distance_percentages_mvcp = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_mvcp, len(levenshtein_results_mvcp))
    _, distance_percentages_hc = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_hc, len(levenshtein_results_hc))
    _, distance_percentages_mv = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_mv, len(levenshtein_results_mv))
    
    return distance_percentages_mvcp, distance_percentages_hc, distance_percentages_mv, levenshtein_results_mvcp, levenshtein_results_hc, levenshtein_results_mv


def create_threshold_table(data, thresholds=[7, 6, 5]):
    tables = {}
    for res_type, threshold_data in data.items():
        df = pd.DataFrame(threshold_data).transpose()
        df_cumulative = pd.DataFrame({f'>={t}': df[df.columns[df.columns >= t]].sum(axis=1) for t in thresholds})
        tables[res_type] = df_cumulative
    return tables

def write_mv_character_results_to_csv(table, mv_character_results, mv_dist, mv_path_images, gt, save_path, name):
    """
    Writes the contents of mv_character_results, distances, paths, and ground truths to a CSV file.
    Also saves a separate summary CSV file with table data.

    Parameters:
        table (dict): Dictionary containing summary tables (e.g., LR, HR, SR) as pandas DataFrames.
        mv_character_results (dict): Dictionary with main character recognition results.
        mv_dist (dict): Dictionary with Levenshtein distances or similar metrics.
        mv_path_images (list): List of paths or identifiers for each image row.
        gt (list): List of ground truth values for each row.
        save_path (Path or str): Directory path to save the output CSV file.
        name (str): Name of the main output CSV file.
    """
    # Define paths for the main and summary CSV files
    output_path = Path(save_path) / name
    summary_output_path = Path(save_path) / f"summary_{name}"

    # Get categories and indices from the mv_character_results structure
    categories = list(mv_character_results.keys())
    indices = list(mv_character_results[categories[0]].keys())

    # Construct the CSV header
    header = ["path", "Gt"] + [f"{cat}_{idx}" for cat in categories for idx in indices] + \
             [f"#correct_{cat}_{idx}" for cat in categories for idx in indices]

    # Write the main CSV file
    with output_path.open(mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        
        # Determine the number of rows based on the length of lists in mv_character_results
        num_rows = len(next(iter(mv_character_results['lr'].values())))  # Assumes uniform length across all lists
        
        # Write each row to the CSV
        for row_idx in range(num_rows):
            # Construct the row with a single list comprehension for efficient data gathering
            row = [
                str(mv_path_images[row_idx]), str(gt[row_idx]),  # Convert path and gt to strings
                *[mv_character_results[cat][idx][row_idx] for cat in categories for idx in indices],
                *[mv_dist[row_idx][cat][idx] for cat in categories for idx in indices]
            ]
            writer.writerow(row)

    # Write summary tables to a separate CSV file
    with summary_output_path.open(mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        
        # Iterate through each category in the table and write the data
        for cat, data in table.items():
            # Write category section header
            writer.writerow([cat])
            
            # Round data to 1 decimal place
            data = data.round(1)
            
            # Write DataFrame headers and contents to CSV
            writer.writerow([''] + list(data.columns))  # Column headers with an empty leading cell for row indices
            writer.writerows([[idx] + row.tolist() for idx, row in data.iterrows()])  # Each row with index
def align_pred_to_gt(pred, gt):
    pred = pred.strip() if pred is not None else ""
    gt = gt.strip()
    L = len(gt)

    pred_chars = list(pred[:L])
    if len(pred_chars) < L:
        pred_chars += [""] * (L - len(pred_chars))

    gt_chars = list(gt)
    return pred_chars, gt_chars

def calc_xlpsr_score_and_acc(pred, gt):
    pred_chars, gt_chars = align_pred_to_gt(pred, gt)

    score = 0
    for p, g in zip(pred_chars, gt_chars):
        if p == "":
            score += 0
        elif p == g:
            score += 2
        else:
            score += -1

    acc = int("".join(pred_chars) == gt)
    return score, acc

def build_conf_filtered_string(pred, conf, threshold):
    # pred: OCR字串
    # conf: 平均置信度或單一分數；若不是list，就整串共用
    if pred is None:
        return ""

    pred = pred.strip()

    if isinstance(conf, (list, tuple, np.ndarray)):
        out = []
        for i, ch in enumerate(pred):
            c = conf[i] if i < len(conf) else 0.0
            out.append(ch if c >= threshold else "")
        return "".join(out)
    else:
        # 若只有整串平均信心，低於threshold就整串空白
        return pred if float(conf) >= threshold else ""

def write_xlpsr_results_to_csv(rows, save_path, filename="xlpsr_eval.csv"):
    output_path = Path(save_path) / filename
    header = [
        "path", "gt",
        "pred_lr", "pred_sr",
        "conf_lr", "conf_sr",
        "score_lr", "score_sr",
        "acc_lr", "acc_sr"
    ]

    with output_path.open(mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row["path"], row["gt"],
                row["pred_lr"], row["pred_sr"],
                row["conf_lr"], row["conf_sr"],
                row["score_lr"], row["score_sr"],
                row["acc_lr"], row["acc_sr"],
            ])
def test(test_loader, model_sr, model_ocr, save_path):
    if model_sr is not None:
        model_sr.eval()
    model_ocr.eval()

    pbar = tqdm(test_loader, leave=False, desc='test')
    results_path = save_path / Path('imgs')
    results_path.mkdir(parents=True, exist_ok=True)

    timings = []
    rows = []

    # 低信心視為空白，依你需求自己調
    conf_threshold = 0.5

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            imgs_lr = batch['lr'].view(-1, 3, 16, 48)

            start = time.time()
            if model_sr is not None:
                imgs_sr = model_sr(imgs_lr.cuda())
            else:
                imgs_sr = F.interpolate(
                    imgs_lr.cuda(), size=(32, 96), mode='bilinear', align_corners=False
                )
            timings.append((time.time() - start) * 1000)

            # LR baseline：先放大到 SR 同大小再 OCR
            lr_up = F.interpolate(
                imgs_lr,
                size=(imgs_sr.size(2), imgs_sr.size(3)),
                mode='bilinear',
                align_corners=False
            ).cuda()

            preds_dict = {
                'lr': model_ocr.OCR_pred(lr_up),
                'sr': model_ocr.OCR_pred(imgs_sr)
            }

            gt = batch['gt'][0]
            sample_path = Path(batch['name'][0])
            sample_name = sample_path.parent.name

            # 預設拿 threshold=1 的結果（你若要改成別的規則再調）
            pred_lr_raw, conf_lr_raw = preds_dict['lr']
            pred_sr_raw, conf_sr_raw = preds_dict['sr']

            pred_lr = select_most_frequent_string(conf_lr_raw, pred_lr_raw, 1)
            pred_sr = select_most_frequent_string(conf_sr_raw, pred_sr_raw, 1)

            # 取平均信心（若 OCR 回傳的是 list）
            if isinstance(conf_lr_raw, (list, tuple, np.ndarray)):
                conf_lr = float(np.mean(conf_lr_raw)) if len(conf_lr_raw) > 0 else 0.0
            else:
                conf_lr = float(conf_lr_raw)

            if isinstance(conf_sr_raw, (list, tuple, np.ndarray)):
                conf_sr = float(np.mean(conf_sr_raw)) if len(conf_sr_raw) > 0 else 0.0
            else:
                conf_sr = float(conf_sr_raw)

            # 低信心 => 空白 => 0分
            pred_lr_final = build_conf_filtered_string(pred_lr, conf_lr, conf_threshold)
            pred_sr_final = build_conf_filtered_string(pred_sr, conf_sr, conf_threshold)

            score_lr, acc_lr = calc_xlpsr_score_and_acc(pred_lr_final, gt)
            score_sr, acc_sr = calc_xlpsr_score_and_acc(pred_sr_final, gt)

            rows.append({
                "path": sample_name,
                "gt": gt,
                "pred_lr": pred_lr_final,
                "pred_sr": pred_sr_final,
                "conf_lr": round(conf_lr, 4),
                "conf_sr": round(conf_sr, 4),
                "score_lr": score_lr,
                "score_sr": score_sr,
                "acc_lr": acc_lr,
                "acc_sr": acc_sr,
            })

            img_save_path = results_path / sample_path.parent
            img_save_path.mkdir(parents=True, exist_ok=True)

            for i, (img_lr_i, img_sr_i) in enumerate(zip(imgs_lr, imgs_sr)):
                filename_lr = f"lr-{i+1:03}.png"
                filename_sr = f"sr-{i+1:03}.png"

                img_lr_i = T.ToPILImage()(img_lr_i)
                img_sr_i = T.ToPILImage()(img_sr_i.cpu())

                img_lr_i.save(img_save_path / Path(filename_lr))
                img_sr_i.save(img_save_path / Path(filename_sr))

    write_xlpsr_results_to_csv(rows, save_path, "xlpsr_eval.csv")

    mean_score_lr = np.mean([r["score_lr"] for r in rows]) if rows else 0.0
    mean_score_sr = np.mean([r["score_sr"] for r in rows]) if rows else 0.0
    accu_lr = np.mean([r["acc_lr"] for r in rows]) if rows else 0.0
    accu_sr = np.mean([r["acc_sr"] for r in rows]) if rows else 0.0
    mean_time = np.mean(timings) if len(timings) > 0 else 0.0

    print(f"[LR ] mean_score={mean_score_lr:.4f}, accu={accu_lr:.4f}")
    print(f"[SR ] mean_score={mean_score_sr:.4f}, accu={accu_sr:.4f}")
    print(f"[SR ] mean_inference_time_ms={mean_time:.2f}")
 
def main(config_, save_path):
    global config
    config = config_    
         
    # Call the prepare_testing function to set up testing
    test_loader, model_sr, model_ocr = prepare_testing()

    # Call the test function to perform the testing
    test(test_loader, model_sr, model_ocr, save_path)
    

if __name__ == '__main__':            
    # Create an argument parser to parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)

    # Parse the command line arguments
    args = parser.parse_args()
    
    def reset_gpu():
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Reset NVIDIA driver cache (requires nvidia-ml-py)
        try:
            nvmlInit()
            for i in range(torch.cuda.device_count()):
                handle = nvmlDeviceGetHandleByIndex(i)
                nvmlDeviceResetGpuLockedClocks(handle)
            nvmlShutdown()
        except:
            pass  # Fallback if pynvml not installed
    
    # Usage
    reset_gpu()
    
    # Define a function to set random seeds for reproducibility
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set a fixed random seed (for reproducibility)
    setup_seed(1996)
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Create a save_name based on the configuration file and tag
    save_name = Path(args.save)
    if save_name is not None:
        save_name = save_name / Path(args.config.split('/')[-1][:-len('.yaml')])
    if args.tag is not None:
        save_name = Path(str(save_name) + '_' + args.tag)
    
    # Create a save_path directory for saving the test results
    save_path = Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)

    # Call the main function to start the testing process
    main(config, save_path)
