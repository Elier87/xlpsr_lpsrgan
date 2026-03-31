import csv
from pathlib import Path


def align_prediction_to_gt(pred, gt):
    pred = (pred or '').strip()
    gt = (gt or '').strip()
    gt_len = len(gt)

    pred_chars = list(pred[:gt_len])
    if len(pred_chars) < gt_len:
        pred_chars += [''] * (gt_len - len(pred_chars))
    return pred_chars, list(gt)


def calc_sequence_metrics(pred, gt):
    pred_chars, gt_chars = align_prediction_to_gt(pred, gt)

    score = 0
    char_hits = 0
    for pred_ch, gt_ch in zip(pred_chars, gt_chars):
        if pred_ch == '':
            score += 0
        elif pred_ch == gt_ch:
            score += 2
            char_hits += 1
        else:
            score += -1

    return {
        'score': score,
        'exact': int(''.join(pred_chars) == gt),
        'char_acc': char_hits / max(len(gt_chars), 1),
    }


def evaluate_prediction_rows(rows):
    labeled = [row for row in rows if row.get('gt')]
    if not labeled:
        return {}

    metrics = [calc_sequence_metrics(row['license_plate'], row['gt']) for row in labeled]
    return {
        'sequence_acc': float(sum(item['exact'] for item in metrics) / len(metrics)),
        'char_acc': float(sum(item['char_acc'] for item in metrics) / len(metrics)),
        'xlpsr_score': float(sum(item['score'] for item in metrics) / len(metrics)),
    }


def write_prediction_csv(rows, save_path, filename='prediction.csv'):
    output_path = Path(save_path) / filename
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'license_plate'])
        for row in rows:
            writer.writerow([row['sequence_id'], row['license_plate']])
    return output_path


def write_debug_csv(rows, save_path, filename='challenge_debug.csv'):
    output_path = Path(save_path) / filename
    if not rows:
        return output_path

    keys = list(rows[0].keys())
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
