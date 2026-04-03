from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    return Image.fromarray(image)


def save_challenge_debug_image(
    lr_image,
    sr_image,
    output_path,
    gt_text=None,
    lr_pred=None,
    sr_pred=None,
    score_lr=None,
    score_sr=None,
    final_pred=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lr_image = _to_pil(lr_image)
    sr_image = _to_pil(sr_image)
    lr_image = lr_image.resize(sr_image.size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(lr_image)
    axes[0].set_title('LR')
    axes[0].axis('off')
    lr_caption = f'LR OCR: {lr_pred or ""}'
    if score_lr is not None:
        lr_caption += f'\nscore_lr: {score_lr}'
    axes[0].text(
        0.5, -0.12, lr_caption, transform=axes[0].transAxes,
        fontsize=11, ha='center', va='top', color='darkorange'
    )

    axes[1].imshow(sr_image)
    axes[1].set_title('SR')
    axes[1].axis('off')
    sr_caption = f'SR OCR: {sr_pred or ""}'
    if score_sr is not None:
        sr_caption += f'\nscore_sr: {score_sr}'
    if final_pred is not None:
        sr_caption += f'\nfinal: {final_pred}'
    if gt_text is not None:
        sr_caption += f'\nGT: {gt_text}'
    axes[1].text(
        0.5, -0.18, sr_caption, transform=axes[1].transAxes,
        fontsize=11, ha='center', va='top', color='blue'
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
