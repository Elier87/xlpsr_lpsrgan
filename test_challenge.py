import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import models
from models.ocr_adapters import calc_text_score_and_acc
from train import make_dataloader
from utils_challenge import (
    aggregate_sequence_predictions,
    apply_plate_format,
    evaluate_prediction_rows,
    save_challenge_debug_image,
    write_debug_csv,
    write_prediction_csv,
)


def prepare_testing(config):
    test_loader = make_dataloader(config['test_dataset'], tag='test')

    model_sr = None
    if config.get('model') is not None:
        sr_ckpt = torch.load(config['model']['load'], map_location='cpu')
        model_sr, _ = models.make(sr_ckpt['model'], load_model=True)
        model_sr = model_sr.cuda().eval()

    model_ocr = models.make(config['model_ocr']).cuda().eval()
    if hasattr(model_ocr, 'freeze') and config['model_ocr'].get('freeze', True):
        model_ocr.freeze()

    return test_loader, model_sr, model_ocr


def run_sr(model_sr, images, sr_size):
    if model_sr is None:
        return F.interpolate(images, size=sr_size, mode='bilinear', align_corners=False)
    return model_sr(images)


def flatten_confidences(confidences):
    if torch.is_tensor(confidences):
        return confidences.detach().cpu().tolist()
    return [float(item) for item in confidences]


def predict_ocr(model_ocr, images, config):
    kwargs = {}
    if config.get('frlp_conf_threshold') is not None:
        kwargs['confidence_threshold'] = config['frlp_conf_threshold']
    try:
        return model_ocr.predict(images, **kwargs)
    except TypeError:
        return model_ocr.predict(images)


def compute_debug_scores(gt, pred_lr, pred_sr):
    if not gt:
        return None, None
    score_lr, _, _ = calc_text_score_and_acc(pred_lr, gt)
    score_sr, _, _ = calc_text_score_and_acc(pred_sr, gt)
    return score_lr, score_sr


def infer_sequence(model_sr, model_ocr, sample_lr, config):
    sr_size = tuple(config.get('sr_size', [32, 96]))
    decision_strategy = config.get('decision_strategy', 'most_frequent')
    fusion_strategy = config.get('fusion_strategy', 'none')

    if sample_lr.dim() == 3:
        frame_lr = sample_lr.unsqueeze(0).cuda()
        frame_sr = run_sr(model_sr, frame_lr, sr_size)
        lr_up = F.interpolate(frame_lr, size=sr_size, mode='bilinear', align_corners=False)
        lr_pred = predict_ocr(model_ocr, lr_up, config)
        sr_pred = predict_ocr(model_ocr, frame_sr, config)
        final_text = sr_pred['texts'][0]
        final_conf = float(flatten_confidences(sr_pred['confidences'])[0])
        return {
            'pred_lr': lr_pred['texts'][0],
            'conf_lr': float(flatten_confidences(lr_pred['confidences'])[0]),
            'pred_sr_frames': sr_pred['texts'],
            'conf_sr_frames': flatten_confidences(sr_pred['confidences']),
            'pred_final': final_text,
            'conf_final': final_conf,
            'lr_debug_image': lr_up[0].detach().cpu(),
            'sr_debug_image': frame_sr[0].detach().cpu(),
            'sr_debug_pred': sr_pred['texts'][0],
        }

    frame_lr = sample_lr.cuda()
    frame_sr = run_sr(model_sr, frame_lr, sr_size)
    lr_up = F.interpolate(frame_lr, size=sr_size, mode='bilinear', align_corners=False)
    lr_pred = predict_ocr(model_ocr, lr_up, config)
    sr_pred = predict_ocr(model_ocr, frame_sr, config)

    texts = list(sr_pred['texts'])
    confidences = flatten_confidences(sr_pred['confidences'])
    debug_sr_image = frame_sr[0].detach().cpu()
    debug_sr_pred = texts[0] if texts else ''

    if fusion_strategy == 'average':
        fused_sr = frame_sr.mean(dim=0, keepdim=True)
        fused_pred = predict_ocr(model_ocr, fused_sr, config)
        texts.append(fused_pred['texts'][0])
        confidences.append(float(flatten_confidences(fused_pred['confidences'])[0]))
        if float(flatten_confidences(fused_pred['confidences'])[0]) >= max(confidences[:-1] or [0.0]):
            debug_sr_image = fused_sr[0].detach().cpu()
            debug_sr_pred = fused_pred['texts'][0]

    if confidences:
        best_sr_idx = int(torch.tensor(confidences).argmax().item())
        if best_sr_idx < frame_sr.size(0):
            debug_sr_image = frame_sr[best_sr_idx].detach().cpu()
            debug_sr_pred = texts[best_sr_idx]

    lr_confidences = flatten_confidences(lr_pred['confidences'])
    best_lr_idx = int(torch.tensor(lr_confidences).argmax().item()) if lr_confidences else 0

    final_text = aggregate_sequence_predictions(texts, confidences, strategy=decision_strategy)
    final_conf = max(confidences) if confidences else 0.0
    return {
        'pred_lr': aggregate_sequence_predictions(
            list(lr_pred['texts']),
            lr_confidences,
            strategy=decision_strategy,
        ),
        'conf_lr': max(lr_confidences) if lr_pred['texts'] else 0.0,
        'pred_sr_frames': texts,
        'conf_sr_frames': confidences,
        'pred_final': final_text,
        'conf_final': final_conf,
        'lr_debug_image': lr_up[best_lr_idx].detach().cpu(),
        'sr_debug_image': debug_sr_image,
        'sr_debug_pred': debug_sr_pred,
    }


def test(config, save_path):
    test_loader, model_sr, model_ocr = prepare_testing(config)
    save_path.mkdir(parents=True, exist_ok=True)
    debug_img_dir = save_path / 'debug_imgs'
    debug_img_dir.mkdir(parents=True, exist_ok=True)

    use_plate_format = config.get('plate_format', {}).get('enabled', False)
    plate_country = config.get('plate_format', {}).get('country', 'fr')

    rows = []
    pbar = tqdm(test_loader, leave=False, desc='challenge-test')
    with torch.no_grad():
        for batch in pbar:
            batch_lr = batch['lr']
            for sample_idx in range(batch_lr.size(0)):
                result = infer_sequence(model_sr, model_ocr, batch_lr[sample_idx], config)
                sequence_id = batch['sequence_id'][sample_idx]
                gt = batch['gt'][sample_idx] if batch.get('gt') is not None else None
                score_lr, score_sr = compute_debug_scores(gt, result['pred_lr'], result['pred_final'])
                final_text = apply_plate_format(
                    result['pred_final'],
                    enabled=use_plate_format,
                    country=plate_country,
                )

                rows.append(
                    {
                        'sequence_id': sequence_id,
                        'license_plate': final_text,
                        'gt': gt,
                        'pred_lr': result['pred_lr'],
                        'pred_sr_frames': '|'.join(result['pred_sr_frames']),
                        'conf_lr': result['conf_lr'],
                        'conf_final': result['conf_final'],
                        'score_lr': score_lr,
                        'score_sr': score_sr,
                        'final_pred': result['pred_final'],
                        'frame_names': '|'.join(batch['frame_names'][sample_idx]),
                        'quality_scores': '|'.join(
                            f'{score:.4f}' for score in batch['quality_scores'][sample_idx]
                        ),
                    }
                )

                save_challenge_debug_image(
                    result['lr_debug_image'].permute(1, 2, 0).numpy(),
                    result['sr_debug_image'].permute(1, 2, 0).numpy(),
                    debug_img_dir / f'{sequence_id}.png',
                    gt_text=gt,
                    lr_pred=result['pred_lr'],
                    sr_pred=result['sr_debug_pred'],
                    score_lr=score_lr,
                    score_sr=score_sr,
                    final_pred=final_text,
                )

    pred_path = write_prediction_csv(
        rows,
        save_path,
        filename=config.get('prediction_filename', 'prediction.csv'),
    )
    debug_path = write_debug_csv(
        rows,
        save_path,
        filename=config.get('debug_filename', 'challenge_debug.csv'),
    )
    metrics = evaluate_prediction_rows(rows)
    return pred_path, debug_path, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--save', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    pred_path, debug_path, metrics = test(config, Path(args.save))
    print(f'prediction_csv: {pred_path}')
    print(f'debug_csv: {debug_path}')
    if metrics:
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')
