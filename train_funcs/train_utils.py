from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from train_funcs import register


def save_visualized_images(image1, image2, image3, output_path, sr_text='SR', gt_text='GT'):
    sr_size = image2.size
    image1_resized = image1.resize(sr_size)
    image3_resized = image3.resize(sr_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image1_resized)
    axes[0].set_title('LR')
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title('SR')
    axes[1].axis('off')
    axes[1].text(
        0.5, -0.1, f'Pred: {sr_text}', transform=axes[1].transAxes,
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


def _maybe_visualize(batch, sr_batch, config, preds=None):
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
    pred_text = preds[visualize_idx] if preds else 'SR'
    save_visualized_images(image1, image2, image3, output_path, pred_text, gt_text)


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


@register('lpsrgan_pretrain')
def lpsrgan_pretrain(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
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

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        _maybe_visualize(batch, sr_batch, config, preds=preds)
        pbar.set_postfix(_format_postfix(metrics))

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_train')
def lpsrgan_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
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

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        train_losses.append(metrics['total'])
        d_losses.append(loss_d.detach().item())
        _maybe_visualize(batch, sr_batch, config, preds=preds)
        pbar.set_postfix(_format_postfix(metrics, extra={'d': round(d_losses[-1], 4)}))

    return sum(train_losses) / len(train_losses)


@register('lpsrgan_val')
def lpsrgan_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    model_g, model_d = _get_model_parts(model)

    model_g.eval()
    if model_d is not None:
        model_d.eval()

    val_losses = []
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
            _maybe_visualize(batch, sr_batch, config, preds=preds)
            pbar.set_postfix(_format_postfix(metrics))

    return sum(val_losses) / len(val_losses), []
