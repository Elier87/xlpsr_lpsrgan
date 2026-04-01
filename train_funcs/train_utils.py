import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from models.ocr_adapters import calc_text_score_and_acc
from train_funcs import register


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
