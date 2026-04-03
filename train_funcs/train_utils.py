import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from models.ocr_adapters import calc_text_score_and_acc
from train_funcs import register
from utils_challenge import aggregate_sequence_predictions, save_challenge_debug_image


def save_visualized_images(
    image1,
    image2,
    image3,
    output_path,
    lr_text='LR',
    sr_text='SR',
    gt_text='GT',
    score_lr=None,
    score_sr=None,
):
    sr_size = image2.size
    image1_resized = image1.resize(sr_size)
    image3_resized = image3.resize(sr_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image1_resized)
    axes[0].set_title('LR')
    axes[0].axis('off')
    lr_caption = f'LR OCR: {lr_text}'
    if score_lr is not None:
        lr_caption += f'\nscore_lr: {score_lr}'
    axes[0].text(
        0.5, -0.1, lr_caption, transform=axes[0].transAxes,
        fontsize=12, ha='center', va='top', color='darkorange'
    )

    axes[1].imshow(image2)
    axes[1].set_title('SR')
    axes[1].axis('off')
    sr_caption = f'SR OCR: {sr_text}'
    if score_sr is not None:
        sr_caption += f'\nscore_sr: {score_sr}'
    axes[1].text(
        0.5, -0.1, sr_caption, transform=axes[1].transAxes,
        fontsize=12, ha='center', va='top', color='blue'
    )

    axes[2].imshow(image3_resized)
    axes[2].set_title('HR')
    axes[2].axis('off')
    axes[2].text(
        0.5, -0.1, f'GT: {gt_text}', transform=axes[2].transAxes,
        fontsize=12, ha='center', va='top', color='green'
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def _get_model_parts(model):
    if isinstance(model, (tuple, list)):
        model_g = model[0]
        model_d = model[1] if len(model) > 1 else None
    else:
        model_g = model
        model_d = None
    return model_g, model_d


def _get_optimizer_parts(optimizer):
    if isinstance(optimizer, (tuple, list)):
        optimizer_g = optimizer[0]
        optimizer_d = optimizer[1] if len(optimizer) > 1 else None
    else:
        optimizer_g = optimizer
        optimizer_d = None
    return optimizer_g, optimizer_d


def _maybe_visualize(batch, sr_batch, config, preds=None, ocr_model=None):
    visualize_interval = config.get('visualize_interval', 0)
    visualize_idx = config.get('visualize_idx', 0)
    current_step = config.get('_loop_step', 0)

    if not visualize_interval or current_step % visualize_interval != 0:
        return

    save_name = config.get('tag_view')
    if not save_name:
        return

    output_path = Path(f"{config['model']['name']}_{save_name}.png")
    image1 = transforms.ToPILImage()(batch['lr'][visualize_idx].detach().cpu())
    image2 = transforms.ToPILImage()(sr_batch[visualize_idx].detach().cpu())
    image3 = transforms.ToPILImage()(batch['hr'][visualize_idx].detach().cpu())
    gt_text = batch['gt'][visualize_idx]
    lr_text = 'LR'
    sr_text = preds[visualize_idx] if preds else 'SR'

    if ocr_model is not None:
        with torch.no_grad():
            lr_up = F.interpolate(
                batch['lr'][visualize_idx:visualize_idx + 1].cuda(),
                size=(sr_batch.size(2), sr_batch.size(3)),
                mode='bilinear',
                align_corners=False,
            )
            sr_single = sr_batch[visualize_idx:visualize_idx + 1].cuda()
            lr_eval = ocr_model.predict(lr_up)
            sr_eval = ocr_model.predict(sr_single)
            lr_text = lr_eval['texts'][0]
            sr_text = sr_eval['texts'][0]

    save_visualized_images(
        image1,
        image2,
        image3,
        output_path,
        lr_text=lr_text,
        sr_text=sr_text,
        gt_text=gt_text,
    )


def _run_generator_loss(loss_fn, sr_batch, batch, loss_adv=None):
    return loss_fn(sr_batch, batch['hr'].cuda(), batch.get('gt'), loss_adv)


def _format_postfix(metrics, extra=None):
    postfix = {'loss': round(metrics['total'], 4)}
    if 'pixel' in metrics:
        postfix['pixel'] = round(metrics['pixel'], 4)
    if 'perceptual' in metrics:
        postfix['perc'] = round(metrics['perceptual'], 4)
    if extra:
        postfix.update(extra)
    return postfix


def _get_ocr_model(args):
    return args[1] if len(args) > 1 else None


def _get_ocr_optimizer(optimizer):
    if isinstance(optimizer, (tuple, list)) and len(optimizer) > 2:
        return optimizer[2]
    return None


def _get_ocr_input(batch, config):
    input_key = config.get('ocr_input_key', 'hr')
    if input_key not in batch:
        raise KeyError(f'OCR input key "{input_key}" not found in batch')
    return batch[input_key].cuda()


def _normalize_confidences(confidences, batch_size):
    if torch.is_tensor(confidences):
        return confidences.detach().cpu().float().view(-1).tolist()
    if isinstance(confidences, (list, tuple)):
        values = []
        for item in confidences:
            if torch.is_tensor(item):
                values.append(float(item.detach().cpu().float().mean().item()))
            else:
                values.append(float(item))
        return values
    return [float(confidences)] * batch_size


def _ocr_predict(model_ocr, images, config=None):
    kwargs = {}
    if config is not None and config.get('frlp_conf_threshold') is not None:
        kwargs['confidence_threshold'] = config['frlp_conf_threshold']
    try:
        return model_ocr.predict(images, **kwargs)
    except TypeError:
        return model_ocr.predict(images)


def _ocr_evaluate(model_ocr, images, targets, config=None):
    kwargs = {}
    if config is not None and config.get('frlp_conf_threshold') is not None:
        kwargs['confidence_threshold'] = config['frlp_conf_threshold']
    try:
        return model_ocr.evaluate(images, targets, **kwargs)
    except TypeError:
        return model_ocr.evaluate(images, targets)


def _apply_confidence_filter(pred, conf, threshold):
    pred = pred.strip() if pred is not None else ''
    if threshold is None:
        return pred
    if isinstance(conf, (list, tuple)):
        out = []
        for idx, ch in enumerate(pred):
            score = conf[idx] if idx < len(conf) else 0.0
            out.append(ch if float(score) >= threshold else '')
        return ''.join(out)
    return pred if float(conf) >= threshold else ''


def _write_val_ocr_csv(rows, config):
    save_root = config.get('save_path')
    epoch = config.get('epoch')
    if save_root is None or epoch is None or not rows:
        return None

    output_dir = Path(save_root) / 'val_ocr'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'epoch_{int(epoch):03d}.csv'

    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['name', 'gt', 'pred_lr', 'pred_sr', 'conf_lr', 'conf_sr', 'score_lr', 'score_sr'],
            extrasaction='ignore',
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _save_val_visualizations(batch, sr_batch, sample_rows, config, start_idx):
    save_root = config.get('save_path')
    epoch = config.get('epoch')
    if save_root is None or epoch is None:
        return 0

    max_visuals = int(config.get('val_visualize_max_samples', 16))
    if start_idx >= max_visuals:
        return 0

    output_dir = Path(save_root) / 'val_ocr' / f'epoch_{int(epoch):03d}_imgs'
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = max_visuals - start_idx
    save_count = min(len(sample_rows), remaining)

    upsampled_lr = F.interpolate(
        batch['lr'], size=(sr_batch.size(2), sr_batch.size(3)), mode='bilinear', align_corners=False
    )

    for local_idx in range(save_count):
        row = sample_rows[local_idx]
        image1 = transforms.ToPILImage()(upsampled_lr[local_idx].detach().cpu())
        image2 = transforms.ToPILImage()(sr_batch[local_idx].detach().cpu())
        image3 = transforms.ToPILImage()(batch['hr'][local_idx].detach().cpu())
        image_name = str(row['name']).replace('/', '_')
        output_path = output_dir / f'{start_idx + local_idx:02d}_{image_name}.png'
        save_visualized_images(
            image1,
            image2,
            image3,
            output_path,
            lr_text=row['pred_lr'],
            sr_text=row['pred_sr'],
            gt_text=row['gt'],
            score_lr=row['score_lr'],
            score_sr=row['score_sr'],
        )
    return save_count


def _maybe_add_ocr_supervision(loss, metrics, sr_batch, batch, config, ocr_model):
    if ocr_model is None:
        return loss, metrics

    weight = config.get('ocr_supervision_weight', 0.0)
    if weight <= 0.0:
        return loss, metrics

    reference_key = config.get('ocr_reference_key', 'hr')
    reference_images = batch[reference_key].cuda() if reference_key in batch else None
    ocr_term = ocr_model.teacher_loss(
        sr_batch,
        targets=batch.get('gt'),
        reference_images=reference_images,
        mode=config.get('ocr_supervision_mode', 'distill'),
        temperature=config.get('ocr_supervision_temperature', 1.0),
        confidence_weighted=config.get('ocr_confidence_weighted', True),
    )
    loss = loss + weight * ocr_term
    metrics['ocr_teacher'] = float(ocr_term.detach().item())
    metrics['total'] = float(loss.detach().item())
    return loss, metrics


def _upsample_lr_batch(lr_batch, sr_batch, config):
    sr_size = config.get('sr_size')
    if sr_size is None:
        sr_size = (sr_batch.size(2), sr_batch.size(3))
    return F.interpolate(lr_batch, size=tuple(sr_size), mode='bilinear', align_corners=False)


def _compute_challenge_finetune_loss(sr_batch, batch, config, ocr_model):
    if any(gt is None for gt in batch.get('gt', [])):
        raise RuntimeError('Stage 3 challenge finetune requires GT for every sample')

    lr = batch['lr'].cuda()
    lr_up = _upsample_lr_batch(lr, sr_batch, config)

    anchor_weight = float(config.get('challenge_anchor_weight', 0.1))
    ocr_weight = float(config.get('challenge_ocr_weight', 1.0))

    anchor_loss = F.l1_loss(sr_batch, lr_up)
    total = anchor_weight * anchor_loss
    metrics = {
        'anchor': float(anchor_loss.detach().item()),
        'ocr': 0.0,
        'total': float(total.detach().item()),
    }

    if ocr_model is not None and ocr_weight > 0.0:
        if (
            config.get('challenge_require_gt_ocr_loss', True)
            and config.get('model_ocr', {}).get('name') != 'gplpr_trainable_ocr'
        ):
            raise RuntimeError(
                'Stage 3 GT-based OCR finetune requires model_ocr.name == "gplpr_trainable_ocr"'
            )
        ocr_model.eval()
        ocr_loss = ocr_model.training_loss(sr_batch, batch['gt'])
        total = total + ocr_weight * ocr_loss
        metrics['ocr'] = float(ocr_loss.detach().item())
        metrics['total'] = float(total.detach().item())

    return total, metrics, lr_up


def _collect_ocr_report(lr_eval, sr_eval, batch, config):
    score_conf_threshold = config.get('ocr_score_conf_threshold')
    lr_conf_list = _normalize_confidences(lr_eval['confidences'], len(batch['gt']))
    sr_conf_list = _normalize_confidences(sr_eval['confidences'], len(batch['gt']))

    score_lr_samples = []
    score_sr_samples = []
    rows = []
    for sample_idx, gt_text in enumerate(batch['gt']):
        pred_lr = _apply_confidence_filter(
            lr_eval['texts'][sample_idx], lr_conf_list[sample_idx], score_conf_threshold
        )
        pred_sr = _apply_confidence_filter(
            sr_eval['texts'][sample_idx], sr_conf_list[sample_idx], score_conf_threshold
        )
        score_lr, _, _ = calc_text_score_and_acc(pred_lr, gt_text)
        score_sr, _, _ = calc_text_score_and_acc(pred_sr, gt_text)
        score_lr_samples.append(score_lr)
        score_sr_samples.append(score_sr)
        rows.append(
            {
                'name': batch['name'][sample_idx],
                'gt': gt_text,
                'pred_lr': pred_lr,
                'pred_sr': pred_sr,
                'conf_lr': lr_conf_list[sample_idx],
                'conf_sr': sr_conf_list[sample_idx],
                'score_lr': score_lr,
                'score_sr': score_sr,
            }
        )

    report = {
        'ocr_score_sr': float(sr_eval['metrics']['ocr_score']),
        'ocr_acc_sr': float(sr_eval['metrics']['ocr_acc']),
        'ocr_char_acc_sr': float(sr_eval['metrics']['ocr_char_acc']),
        'ocr_conf_sr': float(sr_eval['metrics']['ocr_conf']),
        'ocr_score_lr': float(lr_eval['metrics']['ocr_score']),
        'ocr_acc_lr': float(lr_eval['metrics']['ocr_acc']),
        'ocr_char_acc_lr': float(lr_eval['metrics']['ocr_char_acc']),
        'mean_score_lr': float(sum(score_lr_samples) / max(len(score_lr_samples), 1)),
        'mean_score_sr': float(sum(score_sr_samples) / max(len(score_sr_samples), 1)),
    }
    return report, rows


def _flatten_challenge_lr(lr_batch):
    if lr_batch.dim() == 5:
        batch_size, num_frames, channels, height, width = lr_batch.shape
        return lr_batch.view(batch_size * num_frames, channels, height, width), batch_size, num_frames
    if lr_batch.dim() == 4:
        batch_size, channels, height, width = lr_batch.shape
        return lr_batch.view(batch_size, channels, height, width), batch_size, 1
    raise ValueError(f'Unsupported challenge lr shape: {tuple(lr_batch.shape)}')


def _repeat_targets(targets, repeat):
    return [target for target in targets for _ in range(repeat)]


def _chunk_list(values, batch_size, num_frames):
    return [values[idx * num_frames:(idx + 1) * num_frames] for idx in range(batch_size)]


def _sequence_consistency_loss(logits, batch_size, num_frames, temperature=1.0):
    if num_frames <= 1:
        return torch.zeros((), device=logits.device)

    steps = logits.size(1)
    vocab = logits.size(2)
    log_probs = F.log_softmax(logits / temperature, dim=-1).view(batch_size, num_frames, steps, vocab)
    probs = log_probs.exp()
    mean_probs = probs.mean(dim=1, keepdim=True).detach()
    kl = F.kl_div(log_probs, mean_probs.expand_as(probs), reduction='none').sum(dim=-1)
    return kl.mean() * (temperature ** 2)


def _compute_stage3_parseq_loss(sr_batch, lr_batch, batch, config, ocr_model, batch_size, num_frames):
    if ocr_model is None:
        raise RuntimeError('Stage 3 parseq finetune requires a frozen OCR teacher')

    lr_up = _upsample_lr_batch(lr_batch, sr_batch, config)

    teacher_weight = float(config.get('challenge_teacher_weight', 1.0))
    sharpness_weight = float(config.get('challenge_sharpness_weight', 0.02))
    consistency_weight = float(config.get('challenge_sequence_consistency_weight', 0.0))
    image_anchor_weight = float(config.get('challenge_image_anchor_weight', 0.0))
    temperature = float(config.get('challenge_teacher_temperature', 1.0))
    confidence_weighted = bool(config.get('challenge_teacher_confidence_weighted', True))

    teacher_loss = ocr_model.teacher_loss(
        sr_batch,
        targets=batch.get('gt'),
        reference_images=lr_up,
        mode=config.get('challenge_teacher_mode', 'distill'),
        temperature=temperature,
        confidence_weighted=confidence_weighted,
    )
    sharpness_loss = ocr_model.teacher_loss(
        sr_batch,
        targets=batch.get('gt'),
        reference_images=None,
        mode='entropy',
        temperature=temperature,
        confidence_weighted=confidence_weighted,
    )

    total = teacher_weight * teacher_loss + sharpness_weight * sharpness_loss
    metrics = {
        'teacher': float(teacher_loss.detach().item()),
        'sharpness': float(sharpness_loss.detach().item()),
        'consistency': 0.0,
        'image_anchor': 0.0,
        'total': float(total.detach().item()),
    }

    if consistency_weight > 0.0 and num_frames > 1:
        student_logits = ocr_model.forward_logits(sr_batch)
        consistency_loss = _sequence_consistency_loss(
            student_logits, batch_size=batch_size, num_frames=num_frames, temperature=temperature
        )
        total = total + consistency_weight * consistency_loss
        metrics['consistency'] = float(consistency_loss.detach().item())

    if image_anchor_weight > 0.0:
        image_anchor = F.l1_loss(sr_batch, lr_up)
        total = total + image_anchor_weight * image_anchor
        metrics['image_anchor'] = float(image_anchor.detach().item())

    metrics['total'] = float(total.detach().item())
    return total, metrics, lr_up


def _collect_stage3_sequence_report(lr_eval, sr_eval, batch, batch_size, num_frames, config):
    flat_targets = _repeat_targets(batch['gt'], num_frames)
    lr_conf_flat = _normalize_confidences(lr_eval['confidences'], len(flat_targets))
    sr_conf_flat = _normalize_confidences(sr_eval['confidences'], len(flat_targets))
    score_conf_threshold = config.get('ocr_score_conf_threshold')

    frame_score_lr = []
    frame_score_sr = []
    for idx, gt_text in enumerate(flat_targets):
        pred_lr = _apply_confidence_filter(
            lr_eval['texts'][idx], lr_conf_flat[idx], score_conf_threshold
        )
        pred_sr = _apply_confidence_filter(
            sr_eval['texts'][idx], sr_conf_flat[idx], score_conf_threshold
        )
        score_lr, _, _ = calc_text_score_and_acc(pred_lr, gt_text)
        score_sr, _, _ = calc_text_score_and_acc(pred_sr, gt_text)
        frame_score_lr.append(score_lr)
        frame_score_sr.append(score_sr)

    lr_texts_by_seq = _chunk_list(lr_eval['texts'], batch_size, num_frames)
    sr_texts_by_seq = _chunk_list(sr_eval['texts'], batch_size, num_frames)
    lr_conf_by_seq = _chunk_list(lr_conf_flat, batch_size, num_frames)
    sr_conf_by_seq = _chunk_list(sr_conf_flat, batch_size, num_frames)

    decision_strategy = config.get('decision_strategy', 'most_frequent')
    seq_rows = []
    sequence_score_lr = []
    sequence_score_sr = []
    sequence_acc_lr = []
    sequence_acc_sr = []
    sequence_char_acc_lr = []
    sequence_char_acc_sr = []

    for sample_idx, gt_text in enumerate(batch['gt']):
        pred_lr_final = aggregate_sequence_predictions(
            lr_texts_by_seq[sample_idx], lr_conf_by_seq[sample_idx], strategy=decision_strategy
        )
        pred_sr_final = aggregate_sequence_predictions(
            sr_texts_by_seq[sample_idx], sr_conf_by_seq[sample_idx], strategy=decision_strategy
        )

        score_lr, exact_lr, char_acc_lr = calc_text_score_and_acc(pred_lr_final, gt_text)
        score_sr, exact_sr, char_acc_sr = calc_text_score_and_acc(pred_sr_final, gt_text)
        sequence_score_lr.append(score_lr)
        sequence_score_sr.append(score_sr)
        sequence_acc_lr.append(exact_lr)
        sequence_acc_sr.append(exact_sr)
        sequence_char_acc_lr.append(char_acc_lr)
        sequence_char_acc_sr.append(char_acc_sr)

        seq_rows.append(
            {
                'name': batch['name'][sample_idx],
                'gt': gt_text,
                'pred_lr': pred_lr_final,
                'pred_sr': pred_sr_final,
                'conf_lr': max(lr_conf_by_seq[sample_idx]) if lr_conf_by_seq[sample_idx] else 0.0,
                'conf_sr': max(sr_conf_by_seq[sample_idx]) if sr_conf_by_seq[sample_idx] else 0.0,
                'score_lr': score_lr,
                'score_sr': score_sr,
            }
        )

    report = {
        'ocr_score_sr': float(sr_eval['metrics']['ocr_score']),
        'ocr_acc_sr': float(sr_eval['metrics']['ocr_acc']),
        'ocr_char_acc_sr': float(sr_eval['metrics']['ocr_char_acc']),
        'ocr_conf_sr': float(sr_eval['metrics']['ocr_conf']),
        'ocr_score_lr': float(lr_eval['metrics']['ocr_score']),
        'ocr_acc_lr': float(lr_eval['metrics']['ocr_acc']),
        'ocr_char_acc_lr': float(lr_eval['metrics']['ocr_char_acc']),
        'mean_score_lr': float(sum(frame_score_lr) / max(len(frame_score_lr), 1)),
        'mean_score_sr': float(sum(frame_score_sr) / max(len(frame_score_sr), 1)),
        'sequence_score_lr': float(sum(sequence_score_lr) / max(len(sequence_score_lr), 1)),
        'sequence_score_sr': float(sum(sequence_score_sr) / max(len(sequence_score_sr), 1)),
        'sequence_acc_lr': float(sum(sequence_acc_lr) / max(len(sequence_acc_lr), 1)),
        'sequence_acc_sr': float(sum(sequence_acc_sr) / max(len(sequence_acc_sr), 1)),
        'sequence_char_acc_lr': float(sum(sequence_char_acc_lr) / max(len(sequence_char_acc_lr), 1)),
        'sequence_char_acc_sr': float(sum(sequence_char_acc_sr) / max(len(sequence_char_acc_sr), 1)),
    }
    return report, seq_rows


def _save_stage3_val_visualizations(batch, lr_up, sr_batch, sample_rows, config, start_idx, num_frames):
    save_root = config.get('save_path')
    epoch = config.get('epoch')
    if save_root is None or epoch is None:
        return 0

    max_visuals = int(config.get('val_visualize_max_samples', 16))
    if start_idx >= max_visuals:
        return 0

    output_dir = Path(save_root) / 'val_ocr' / f'epoch_{int(epoch):03d}_imgs'
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = max_visuals - start_idx
    save_count = min(len(sample_rows), remaining)

    for local_idx in range(save_count):
        row = sample_rows[local_idx]
        frame_idx = local_idx * num_frames
        lr_image = transforms.ToPILImage()(lr_up[frame_idx].detach().cpu())
        sr_image = transforms.ToPILImage()(sr_batch[frame_idx].detach().cpu())
        image_name = str(row['name']).replace('/', '_')
        output_path = output_dir / f'{start_idx + local_idx:02d}_{image_name}.png'
        save_challenge_debug_image(
            lr_image,
            sr_image,
            output_path,
            gt_text=row['gt'],
            lr_pred=row['pred_lr'],
            sr_pred=row['pred_sr'],
            score_lr=row['score_lr'],
            score_sr=row['score_sr'],
            final_pred=row.get('final_pred'),
        )
    return save_count


def _maybe_visualize_stage3(batch, lr_up, sr_batch, config, sample_rows=None, num_frames=1):
    visualize_interval = config.get('visualize_interval', 0)
    current_step = config.get('_loop_step', 0)
    if not visualize_interval or current_step % visualize_interval != 0:
        return

    save_name = config.get('tag_view')
    if not save_name:
        return

    output_path = Path(f"{config['model']['name']}_{save_name}.png")
    row = sample_rows[0] if sample_rows else None
    lr_image = transforms.ToPILImage()(lr_up[0].detach().cpu())
    sr_image = transforms.ToPILImage()(sr_batch[0].detach().cpu())
    save_challenge_debug_image(
        lr_image,
        sr_image,
        output_path,
        gt_text=row.get('gt') if row else batch['gt'][0],
        lr_pred=row.get('pred_lr') if row else None,
        sr_pred=row.get('pred_sr') if row else None,
        score_lr=row.get('score_lr') if row else None,
        score_sr=row.get('score_sr') if row else None,
        final_pred=row.get('final_pred') if row else None,
    )


def _collect_stage3_frlp_report(lr_outputs, sr_outputs, batch, batch_size, num_frames, config):
    flat_targets = _repeat_targets(batch['gt'], num_frames)
    lr_texts = list(lr_outputs['texts'])
    sr_texts = list(sr_outputs['texts'])
    lr_conf_flat = _normalize_confidences(lr_outputs['confidences'], len(flat_targets))
    sr_conf_flat = _normalize_confidences(sr_outputs['confidences'], len(flat_targets))

    frame_score_lr = []
    frame_score_sr = []
    frame_exact_lr = []
    frame_exact_sr = []
    frame_char_acc_lr = []
    frame_char_acc_sr = []

    for idx, gt_text in enumerate(flat_targets):
        score_lr, exact_lr, char_acc_lr = calc_text_score_and_acc(lr_texts[idx], gt_text)
        score_sr, exact_sr, char_acc_sr = calc_text_score_and_acc(sr_texts[idx], gt_text)
        frame_score_lr.append(score_lr)
        frame_score_sr.append(score_sr)
        frame_exact_lr.append(exact_lr)
        frame_exact_sr.append(exact_sr)
        frame_char_acc_lr.append(char_acc_lr)
        frame_char_acc_sr.append(char_acc_sr)

    lr_texts_by_seq = _chunk_list(lr_texts, batch_size, num_frames)
    sr_texts_by_seq = _chunk_list(sr_texts, batch_size, num_frames)
    lr_conf_by_seq = _chunk_list(lr_conf_flat, batch_size, num_frames)
    sr_conf_by_seq = _chunk_list(sr_conf_flat, batch_size, num_frames)

    decision_strategy = config.get('decision_strategy', 'most_frequent')
    rows = []
    sequence_score_lr = []
    sequence_score_sr = []
    sequence_exact_lr = []
    sequence_exact_sr = []
    sequence_char_acc_lr = []
    sequence_char_acc_sr = []

    for sample_idx, gt_text in enumerate(batch['gt']):
        pred_lr_final = aggregate_sequence_predictions(
            lr_texts_by_seq[sample_idx], lr_conf_by_seq[sample_idx], strategy=decision_strategy
        )
        pred_sr_final = aggregate_sequence_predictions(
            sr_texts_by_seq[sample_idx], sr_conf_by_seq[sample_idx], strategy=decision_strategy
        )

        score_lr, exact_lr, char_acc_lr = calc_text_score_and_acc(pred_lr_final, gt_text)
        score_sr, exact_sr, char_acc_sr = calc_text_score_and_acc(pred_sr_final, gt_text)
        sequence_score_lr.append(score_lr)
        sequence_score_sr.append(score_sr)
        sequence_exact_lr.append(exact_lr)
        sequence_exact_sr.append(exact_sr)
        sequence_char_acc_lr.append(char_acc_lr)
        sequence_char_acc_sr.append(char_acc_sr)

        rows.append(
            {
                'name': batch['name'][sample_idx],
                'gt': gt_text,
                'pred_lr': pred_lr_final,
                'pred_sr': pred_sr_final,
                'conf_lr': max(lr_conf_by_seq[sample_idx]) if lr_conf_by_seq[sample_idx] else 0.0,
                'conf_sr': max(sr_conf_by_seq[sample_idx]) if sr_conf_by_seq[sample_idx] else 0.0,
                'score_lr': score_lr,
                'score_sr': score_sr,
                'pred_lr_frames': '|'.join(lr_texts_by_seq[sample_idx]),
                'pred_sr_frames': '|'.join(sr_texts_by_seq[sample_idx]),
                'final_pred': pred_sr_final,
            }
        )

    report = {
        'avg_score_lr': float(sum(sequence_score_lr) / max(len(sequence_score_lr), 1)),
        'avg_score_sr': float(sum(sequence_score_sr) / max(len(sequence_score_sr), 1)),
        'exact_acc_lr': float(sum(sequence_exact_lr) / max(len(sequence_exact_lr), 1)),
        'exact_acc_sr': float(sum(sequence_exact_sr) / max(len(sequence_exact_sr), 1)),
        'char_acc_lr': float(sum(sequence_char_acc_lr) / max(len(sequence_char_acc_lr), 1)),
        'char_acc_sr': float(sum(sequence_char_acc_sr) / max(len(sequence_char_acc_sr), 1)),
        'sequence_score_lr': float(sum(sequence_score_lr) / max(len(sequence_score_lr), 1)),
        'sequence_score_sr': float(sum(sequence_score_sr) / max(len(sequence_score_sr), 1)),
        'frame_score_lr': float(sum(frame_score_lr) / max(len(frame_score_lr), 1)),
        'frame_score_sr': float(sum(frame_score_sr) / max(len(frame_score_sr), 1)),
        'frame_exact_acc_lr': float(sum(frame_exact_lr) / max(len(frame_exact_lr), 1)),
        'frame_exact_acc_sr': float(sum(frame_exact_sr) / max(len(frame_exact_sr), 1)),
        'frame_char_acc_lr': float(sum(frame_char_acc_lr) / max(len(frame_char_acc_lr), 1)),
        'frame_char_acc_sr': float(sum(frame_char_acc_sr) / max(len(frame_char_acc_sr), 1)),
        'conf_lr': float(sum(lr_conf_flat) / max(len(lr_conf_flat), 1)),
        'conf_sr': float(sum(sr_conf_flat) / max(len(sr_conf_flat), 1)),
    }
    return report, rows


@register('lpsrgan_pretrain')
def lpsrgan_pretrain(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)
    optimizer_g, _ = _get_optimizer_parts(optimizer)

    model_g.train()
    if model_d is not None:
        model_d.eval()

    train_losses = []
    pbar = tqdm(train_loader, leave=False, desc='pretrain')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        sr_batch = model_g(batch['lr'].cuda())
        loss_g, metrics, preds = _run_generator_loss(loss_fn, sr_batch, batch, loss_adv=None)
        loss_g, metrics = _maybe_add_ocr_supervision(
            loss_g, metrics, sr_batch, batch, config, ocr_model
        )

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        _maybe_visualize(batch, sr_batch, config, preds=preds, ocr_model=ocr_model)
        pbar.set_postfix(_format_postfix(metrics))

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_train')
def lpsrgan_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)
    optimizer_g, optimizer_d = _get_optimizer_parts(optimizer)

    if model_d is None or optimizer_d is None:
        raise RuntimeError('GAN training requires both generator/discriminator and two optimizers')

    model_g.train()
    model_d.train()

    train_losses = []
    d_losses = []
    pbar = tqdm(train_loader, leave=False, desc='train')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        lr = batch['lr'].cuda()
        hr = batch['hr'].cuda()

        fake = model_g(lr)
        real_pred = model_d(hr)
        fake_pred = model_d(fake.detach())
        loss_d = 0.5 * (torch.mean((real_pred - 1) ** 2) + torch.mean(fake_pred ** 2))

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        sr_batch = model_g(lr)
        fake_pred_for_g = model_d(sr_batch)
        loss_adv = torch.mean((fake_pred_for_g - 1) ** 2)
        loss_g, metrics, preds = _run_generator_loss(loss_fn, sr_batch, batch, loss_adv=loss_adv)
        loss_g, metrics = _maybe_add_ocr_supervision(
            loss_g, metrics, sr_batch, batch, config, ocr_model
        )

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        d_losses.append(loss_d.detach().item())
        _maybe_visualize(batch, sr_batch, config, preds=preds, ocr_model=ocr_model)
        pbar.set_postfix(_format_postfix(metrics, extra={'d': round(d_losses[-1], 4)}))

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_val')
def lpsrgan_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)

    model_g.eval()
    if model_d is not None:
        model_d.eval()

    val_losses = []
    ocr_sr_scores = []
    ocr_sr_accs = []
    ocr_sr_char_accs = []
    ocr_sr_confs = []
    ocr_lr_scores = []
    ocr_lr_accs = []
    ocr_lr_char_accs = []
    score_lr_samples = []
    score_sr_samples = []
    epoch_rows = []
    saved_visuals = 0
    score_conf_threshold = config.get('ocr_score_conf_threshold')
    pbar = tqdm(val_loader, leave=False, desc='val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            config['_loop_step'] = idx + 1
            lr = batch['lr'].cuda()
            hr = batch['hr'].cuda()
            sr_batch = model_g(lr)

            loss_adv = None
            if config.get('training_stage', 'gan') == 'gan' and model_d is not None:
                fake_pred = model_d(sr_batch)
                loss_adv = torch.mean((fake_pred - 1) ** 2)

            loss_g, metrics, preds = _run_generator_loss(loss_fn, sr_batch, batch, loss_adv=loss_adv)
            val_losses.append(metrics['total'])

            if ocr_model is not None:
                lr_up = F.interpolate(
                    batch['lr'].cuda(),
                    size=(sr_batch.size(2), sr_batch.size(3)),
                    mode='bilinear',
                    align_corners=False,
                )
                sr_eval = ocr_model.evaluate(sr_batch, batch['gt'])
                lr_eval = ocr_model.evaluate(lr_up, batch['gt'])

                ocr_sr_scores.append(sr_eval['metrics']['ocr_score'])
                ocr_sr_accs.append(sr_eval['metrics']['ocr_acc'])
                ocr_sr_char_accs.append(sr_eval['metrics']['ocr_char_acc'])
                ocr_sr_confs.append(sr_eval['metrics']['ocr_conf'])
                ocr_lr_scores.append(lr_eval['metrics']['ocr_score'])
                ocr_lr_accs.append(lr_eval['metrics']['ocr_acc'])
                ocr_lr_char_accs.append(lr_eval['metrics']['ocr_char_acc'])

                lr_conf_list = _normalize_confidences(lr_eval['confidences'], len(batch['gt']))
                sr_conf_list = _normalize_confidences(sr_eval['confidences'], len(batch['gt']))
                batch_rows = []
                for sample_idx, gt_text in enumerate(batch['gt']):
                    pred_lr_raw = lr_eval['texts'][sample_idx]
                    pred_sr_raw = sr_eval['texts'][sample_idx]
                    conf_lr = lr_conf_list[sample_idx]
                    conf_sr = sr_conf_list[sample_idx]

                    pred_lr = _apply_confidence_filter(
                        pred_lr_raw, conf_lr, score_conf_threshold
                    )
                    pred_sr = _apply_confidence_filter(
                        pred_sr_raw, conf_sr, score_conf_threshold
                    )
                    score_lr, _, _ = calc_text_score_and_acc(pred_lr, gt_text)
                    score_sr, _, _ = calc_text_score_and_acc(pred_sr, gt_text)

                    score_lr_samples.append(score_lr)
                    score_sr_samples.append(score_sr)
                    batch_rows.append(
                        {
                            'name': batch['name'][sample_idx],
                            'gt': gt_text,
                            'pred_lr': pred_lr,
                            'pred_sr': pred_sr,
                            'conf_lr': conf_lr,
                            'conf_sr': conf_sr,
                            'score_lr': score_lr,
                            'score_sr': score_sr,
                        }
                    )

                epoch_rows.extend(batch_rows)
                saved_visuals += _save_val_visualizations(
                    batch, sr_batch, batch_rows, config, saved_visuals
                )

            _maybe_visualize(batch, sr_batch, config, preds=preds, ocr_model=ocr_model)
            postfix_extra = None
            if ocr_model is not None and ocr_sr_scores:
                postfix_extra = {
                    'ocr_sr': round(ocr_sr_scores[-1], 4),
                    'ocr_lr': round(ocr_lr_scores[-1], 4),
                    'score_sr': round(score_sr_samples[-1], 4),
                    'score_lr': round(score_lr_samples[-1], 4),
                }
            pbar.set_postfix(_format_postfix(metrics, extra=postfix_extra))

    report = {}
    if ocr_sr_scores:
        report = {
            'ocr_score_sr': float(sum(ocr_sr_scores) / len(ocr_sr_scores)),
            'ocr_acc_sr': float(sum(ocr_sr_accs) / len(ocr_sr_accs)),
            'ocr_char_acc_sr': float(sum(ocr_sr_char_accs) / len(ocr_sr_char_accs)),
            'ocr_conf_sr': float(sum(ocr_sr_confs) / len(ocr_sr_confs)),
            'ocr_score_lr': float(sum(ocr_lr_scores) / len(ocr_lr_scores)),
            'ocr_acc_lr': float(sum(ocr_lr_accs) / len(ocr_lr_accs)),
            'ocr_char_acc_lr': float(sum(ocr_lr_char_accs) / len(ocr_lr_char_accs)),
            'mean_score_lr': float(sum(score_lr_samples) / len(score_lr_samples)),
            'mean_score_sr': float(sum(score_sr_samples) / len(score_sr_samples)),
        }
        _write_val_ocr_csv(epoch_rows, config)

    return sum(val_losses) / len(val_losses), report


@register('ocr_train')
def ocr_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    model_g, model_d = _get_model_parts(model)
    optimizer_g, _ = _get_optimizer_parts(optimizer)

    if model_d is not None:
        raise RuntimeError('ocr_train expects a single trainable OCR model')

    model_g.train()
    train_losses = []
    pbar = tqdm(train_loader, leave=False, desc='ocr-train')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        images = _get_ocr_input(batch, config)
        loss = model_g.training_loss(images, batch['gt'])

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        train_losses.append(float(loss.detach().item()))
        pbar.set_postfix({'loss': round(train_losses[-1], 4)})

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_challenge_finetune')
def lpsrgan_challenge_finetune(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)
    optimizer_g, _ = _get_optimizer_parts(optimizer)

    model_g.train()
    if model_d is not None:
        model_d.eval()

    train_losses = []
    pbar = tqdm(train_loader, leave=False, desc='challenge-ft')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        lr = batch['lr'].cuda()
        sr_batch = model_g(lr)
        loss, metrics, _ = _compute_challenge_finetune_loss(sr_batch, batch, config, ocr_model)

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        pbar.set_postfix(
            {
                'loss': round(metrics['total'], 4),
                'ocr': round(metrics['ocr'], 4),
                'anchor': round(metrics['anchor'], 4),
            }
        )

    return sum(train_losses) / len(train_losses)


@register('challenge_finetune_train')
def challenge_finetune_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)
    optimizer_g, _ = _get_optimizer_parts(optimizer)

    model_g.train()
    if model_d is not None:
        model_d.eval()
    if ocr_model is not None:
        ocr_model.eval()

    train_losses = []
    pbar = tqdm(train_loader, leave=False, desc='stage3-train')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        lr_flat, batch_size, num_frames = _flatten_challenge_lr(batch['lr'].cuda())
        sr_batch = model_g(lr_flat)
        loss, metrics, _ = _compute_stage3_parseq_loss(
            sr_batch, lr_flat, batch, config, ocr_model, batch_size, num_frames
        )

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        pbar.set_postfix(
            {
                'loss': round(metrics['total'], 4),
                'teacher': round(metrics['teacher'], 4),
                'sharp': round(metrics['sharpness'], 4),
                'cons': round(metrics['consistency'], 4),
            }
        )

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_challenge_val')
def lpsrgan_challenge_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)

    model_g.eval()
    if model_d is not None:
        model_d.eval()
    if ocr_model is not None:
        ocr_model.eval()

    val_losses = []
    anchor_losses = []
    ocr_losses = []
    epoch_rows = []
    reports = []
    pbar = tqdm(val_loader, leave=False, desc='challenge-val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            config['_loop_step'] = idx + 1
            lr = batch['lr'].cuda()
            sr_batch = model_g(lr)
            loss, metrics, lr_up = _compute_challenge_finetune_loss(
                sr_batch, batch, config, ocr_model
            )

            val_losses.append(metrics['total'])
            anchor_losses.append(metrics['anchor'])
            ocr_losses.append(metrics['ocr'])

            if ocr_model is not None:
                sr_eval = ocr_model.evaluate(sr_batch, batch['gt'])
                lr_eval = ocr_model.evaluate(lr_up, batch['gt'])
                batch_report, batch_rows = _collect_ocr_report(lr_eval, sr_eval, batch, config)
                reports.append(batch_report)
                epoch_rows.extend(batch_rows)
                postfix = {
                    'loss': round(metrics['total'], 4),
                    'ocr_sr': round(batch_report['ocr_score_sr'], 4),
                    'score_sr': round(batch_report['mean_score_sr'], 4),
                }
            else:
                postfix = {
                    'loss': round(metrics['total'], 4),
                    'anchor': round(metrics['anchor'], 4),
                }
            pbar.set_postfix(postfix)

    report = {
        'challenge_anchor': float(sum(anchor_losses) / len(anchor_losses)),
        'challenge_ocr_loss': float(sum(ocr_losses) / len(ocr_losses)),
    }
    if reports:
        keys = reports[0].keys()
        for key in keys:
            report[key] = float(sum(item[key] for item in reports) / len(reports))
        _write_val_ocr_csv(epoch_rows, config)

    return sum(val_losses) / len(val_losses), report


@register('challenge_finetune_val')
def challenge_finetune_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)

    model_g.eval()
    if model_d is not None:
        model_d.eval()
    if ocr_model is not None:
        ocr_model.eval()

    val_losses = []
    teacher_losses = []
    sharpness_losses = []
    consistency_losses = []
    image_anchor_losses = []
    reports = []
    epoch_rows = []
    pbar = tqdm(val_loader, leave=False, desc='stage3-val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            config['_loop_step'] = idx + 1
            lr_flat, batch_size, num_frames = _flatten_challenge_lr(batch['lr'].cuda())
            sr_batch = model_g(lr_flat)
            loss, metrics, lr_up = _compute_stage3_parseq_loss(
                sr_batch, lr_flat, batch, config, ocr_model, batch_size, num_frames
            )

            val_losses.append(metrics['total'])
            teacher_losses.append(metrics['teacher'])
            sharpness_losses.append(metrics['sharpness'])
            consistency_losses.append(metrics['consistency'])
            image_anchor_losses.append(metrics['image_anchor'])

            if ocr_model is not None:
                flat_targets = _repeat_targets(batch['gt'], num_frames)
                sr_eval = ocr_model.evaluate(sr_batch, flat_targets)
                lr_eval = ocr_model.evaluate(lr_up, flat_targets)
                batch_report, batch_rows = _collect_stage3_sequence_report(
                    lr_eval, sr_eval, batch, batch_size, num_frames, config
                )
                reports.append(batch_report)
                epoch_rows.extend(batch_rows)
                postfix = {
                    'loss': round(metrics['total'], 4),
                    'score_sr': round(batch_report['mean_score_sr'], 4),
                    'seq_sr': round(batch_report['sequence_score_sr'], 4),
                }
            else:
                postfix = {'loss': round(metrics['total'], 4)}
            pbar.set_postfix(postfix)

    report = {
        'challenge_teacher': float(sum(teacher_losses) / max(len(teacher_losses), 1)),
        'challenge_sharpness': float(sum(sharpness_losses) / max(len(sharpness_losses), 1)),
        'challenge_consistency': float(sum(consistency_losses) / max(len(consistency_losses), 1)),
        'challenge_image_anchor': float(sum(image_anchor_losses) / max(len(image_anchor_losses), 1)),
    }
    if reports:
        keys = reports[0].keys()
        for key in keys:
            report[key] = float(sum(item[key] for item in reports) / len(reports))
        _write_val_ocr_csv(epoch_rows, config)

    return sum(val_losses) / len(val_losses), report


@register('challenge_frlp_head_train')
def challenge_frlp_head_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)
    optimizer_g, _ = _get_optimizer_parts(optimizer)
    optimizer_ocr = _get_ocr_optimizer(optimizer)

    if ocr_model is None or not hasattr(ocr_model, 'compute_task_loss'):
        raise RuntimeError('challenge_frlp_head_train requires model_ocr=parseq_frlp_head')
    if optimizer_ocr is None:
        raise RuntimeError('Stage 3 FRLP head training requires optimizer_ocr')

    model_g.train()
    if model_d is not None:
        model_d.eval()
    ocr_model.train()

    train_losses = []
    token_losses = []
    position_losses = []
    length_losses = []
    image_reg_losses = []
    report_sums = {}
    report_count = 0

    pbar = tqdm(train_loader, leave=False, desc='stage3-frlp-train')
    for idx, batch in enumerate(pbar):
        config['_loop_step'] = idx + 1
        lr_flat, batch_size, num_frames = _flatten_challenge_lr(batch['lr'].cuda())
        flat_targets = _repeat_targets(batch['gt'], num_frames)
        sr_batch = model_g(lr_flat)
        lr_up = _upsample_lr_batch(lr_flat, sr_batch, config)

        loss, metrics, sr_outputs = ocr_model.compute_task_loss(
            sr_batch,
            flat_targets,
            token_ce_weight=float(config.get('frlp_token_ce_weight', 1.0)),
            position_ce_weight=float(config.get('frlp_position_ce_weight', 0.2)),
            length_weight=float(config.get('frlp_length_weight', 0.2)),
            score_weight=float(config.get('frlp_score_weight', 0.0)),
            confidence_threshold=config.get('frlp_conf_threshold'),
        )

        image_reg_weight = float(config.get('optional_image_regularization_weight', 0.0))
        image_reg_loss = torch.zeros((), device=sr_batch.device)
        if image_reg_weight > 0.0:
            image_reg_loss = F.l1_loss(sr_batch, lr_up)
            loss = loss + image_reg_weight * image_reg_loss

        optimizer_g.zero_grad()
        optimizer_ocr.zero_grad()
        loss.backward()
        optimizer_g.step()
        optimizer_ocr.step()

        with torch.no_grad():
            lr_outputs = _ocr_predict(ocr_model, lr_up, config)
        batch_report, batch_rows = _collect_stage3_frlp_report(
            lr_outputs, sr_outputs, batch, batch_size, num_frames, config
        )
        _maybe_visualize_stage3(
            batch, lr_up, sr_batch, config, sample_rows=batch_rows, num_frames=num_frames
        )

        metrics['total'] = float(loss.detach().item())
        metrics['image_reg'] = float(image_reg_loss.detach().item())
        train_losses.append(metrics['total'])
        token_losses.append(metrics['token_ce'])
        position_losses.append(metrics['position_ce'])
        length_losses.append(metrics['length_ce'])
        image_reg_losses.append(metrics['image_reg'])
        for key, value in batch_report.items():
            report_sums[key] = report_sums.get(key, 0.0) + float(value)
        report_count += 1

        pbar.set_postfix(
            {
                'loss': round(metrics['total'], 4),
                'score_sr': round(batch_report['avg_score_sr'], 4),
                'score_lr': round(batch_report['avg_score_lr'], 4),
                'tok': round(metrics['token_ce'], 4),
            }
        )

    report = {
        'avg_loss': float(sum(train_losses) / max(len(train_losses), 1)),
        'avg_score_sr': float(report_sums.get('avg_score_sr', 0.0) / max(report_count, 1)),
        'avg_score_lr': float(report_sums.get('avg_score_lr', 0.0) / max(report_count, 1)),
        'exact_acc_sr': float(report_sums.get('exact_acc_sr', 0.0) / max(report_count, 1)),
        'char_acc_sr': float(report_sums.get('char_acc_sr', 0.0) / max(report_count, 1)),
        'sequence_score_sr': float(report_sums.get('sequence_score_sr', 0.0) / max(report_count, 1)),
        'token_ce': float(sum(token_losses) / max(len(token_losses), 1)),
        'position_ce': float(sum(position_losses) / max(len(position_losses), 1)),
        'length_ce': float(sum(length_losses) / max(len(length_losses), 1)),
        'image_reg': float(sum(image_reg_losses) / max(len(image_reg_losses), 1)),
    }
    return report['avg_loss'], report


@register('challenge_frlp_head_val')
def challenge_frlp_head_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    ocr_model = _get_ocr_model(args)
    model_g, model_d = _get_model_parts(model)

    if ocr_model is None or not hasattr(ocr_model, 'compute_task_loss'):
        raise RuntimeError('challenge_frlp_head_val requires model_ocr=parseq_frlp_head')

    model_g.eval()
    if model_d is not None:
        model_d.eval()
    ocr_model.eval()

    val_losses = []
    token_losses = []
    position_losses = []
    length_losses = []
    image_reg_losses = []
    report_sums = {}
    report_count = 0
    epoch_rows = []
    saved_visuals = 0

    pbar = tqdm(val_loader, leave=False, desc='stage3-frlp-val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            config['_loop_step'] = idx + 1
            lr_flat, batch_size, num_frames = _flatten_challenge_lr(batch['lr'].cuda())
            flat_targets = _repeat_targets(batch['gt'], num_frames)
            sr_batch = model_g(lr_flat)
            lr_up = _upsample_lr_batch(lr_flat, sr_batch, config)

            loss, metrics, sr_outputs = ocr_model.compute_task_loss(
                sr_batch,
                flat_targets,
                token_ce_weight=float(config.get('frlp_token_ce_weight', 1.0)),
                position_ce_weight=float(config.get('frlp_position_ce_weight', 0.2)),
                length_weight=float(config.get('frlp_length_weight', 0.2)),
                score_weight=float(config.get('frlp_score_weight', 0.0)),
                confidence_threshold=config.get('frlp_conf_threshold'),
            )

            image_reg_weight = float(config.get('optional_image_regularization_weight', 0.0))
            image_reg_loss = torch.zeros((), device=sr_batch.device)
            if image_reg_weight > 0.0:
                image_reg_loss = F.l1_loss(sr_batch, lr_up)
                loss = loss + image_reg_weight * image_reg_loss

            lr_outputs = _ocr_predict(ocr_model, lr_up, config)
            batch_report, batch_rows = _collect_stage3_frlp_report(
                lr_outputs, sr_outputs, batch, batch_size, num_frames, config
            )

            val_losses.append(float(loss.detach().item()))
            token_losses.append(metrics['token_ce'])
            position_losses.append(metrics['position_ce'])
            length_losses.append(metrics['length_ce'])
            image_reg_losses.append(float(image_reg_loss.detach().item()))
            for key, value in batch_report.items():
                report_sums[key] = report_sums.get(key, 0.0) + float(value)
            report_count += 1
            epoch_rows.extend(batch_rows)
            saved_visuals += _save_stage3_val_visualizations(
                batch, lr_up, sr_batch, batch_rows, config, saved_visuals, num_frames
            )

            pbar.set_postfix(
                {
                    'loss': round(val_losses[-1], 4),
                    'score_sr': round(batch_report['avg_score_sr'], 4),
                    'score_lr': round(batch_report['avg_score_lr'], 4),
                }
            )

    _write_val_ocr_csv(epoch_rows, config)
    report = {
        'avg_loss': float(sum(val_losses) / max(len(val_losses), 1)),
        'avg_score_sr': float(report_sums.get('avg_score_sr', 0.0) / max(report_count, 1)),
        'avg_score_lr': float(report_sums.get('avg_score_lr', 0.0) / max(report_count, 1)),
        'exact_acc_sr': float(report_sums.get('exact_acc_sr', 0.0) / max(report_count, 1)),
        'char_acc_sr': float(report_sums.get('char_acc_sr', 0.0) / max(report_count, 1)),
        'sequence_score_sr': float(report_sums.get('sequence_score_sr', 0.0) / max(report_count, 1)),
        'sequence_score_lr': float(report_sums.get('sequence_score_lr', 0.0) / max(report_count, 1)),
        'token_ce': float(sum(token_losses) / max(len(token_losses), 1)),
        'position_ce': float(sum(position_losses) / max(len(position_losses), 1)),
        'length_ce': float(sum(length_losses) / max(len(length_losses), 1)),
        'image_reg': float(sum(image_reg_losses) / max(len(image_reg_losses), 1)),
    }
    return report['avg_loss'], report


@register('ocr_val')
def ocr_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    model_g, model_d = _get_model_parts(model)

    if model_d is not None:
        raise RuntimeError('ocr_val expects a single OCR model')

    model_g.eval()
    val_losses = []
    reports = []
    pbar = tqdm(val_loader, leave=False, desc='ocr-val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            config['_loop_step'] = idx + 1
            images = _get_ocr_input(batch, config)
            loss = model_g.training_loss(images, batch['gt'])
            outputs = model_g.evaluate(images, batch['gt'])

            val_losses.append(float(loss.detach().item()))
            reports.append(outputs['metrics'])
            pbar.set_postfix(
                {
                    'loss': round(val_losses[-1], 4),
                    'ocr': round(outputs['metrics']['ocr_score'], 4),
                }
            )

    report = {}
    if reports:
        keys = reports[0].keys()
        report = {
            key: float(sum(item[key] for item in reports) / len(reports))
            for key in keys
        }
    return sum(val_losses) / len(val_losses), report
