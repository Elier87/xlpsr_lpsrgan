import re
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19

try:
    from torchvision.models import VGG19_Weights
except ImportError:  # torchvision < 0.13
    VGG19_Weights = None

from losses import register


def load_model(path):
    from keras.models import model_from_json

    with open(path + '/model.json', 'r') as f:
        json = f.read()

    model = model_from_json(json)
    model.load_weights(path + '/weights.hdf5')
    return model


def padding(img, min_ratio, max_ratio, color=(0, 0, 0)):
    img_h, img_w = np.shape(img)[:2]
    border_w = 0
    border_h = 0
    ar = float(img_w) / img_h

    if min_ratio <= ar <= max_ratio:
        return img, border_w, border_h

    if ar < min_ratio:
        while ar < min_ratio:
            border_w += 1
            ar = float(img_w + border_w) / (img_h + border_h)
    else:
        while ar > max_ratio:
            border_h += 1
            ar = float(img_w) / (img_h + border_h)

    border_w //= 2
    border_h //= 2

    img = cv2.copyMakeBorder(
        img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=color
    )
    return img, border_w, border_h


@register('noopLoss')
class NoOpLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        device = 'cpu'
        for arg in args:
            if torch.is_tensor(arg):
                device = arg.device
                break
        zero = torch.zeros((), device=device)
        return zero, {'total': 0.0}, None


@register('lpsrganLoss')
class LPSRGANLoss(nn.Module):
    def __init__(
        self,
        pixel_weight=1.0,
        perceptual_weight=0.1,
        adversarial_weight=0.005,
        use_perceptual=True,
        use_ocr=False,
        ocr_weight=0.0,
        load=None,
        weight=None,
        size_average=None,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.use_perceptual = use_perceptual
        self.use_ocr = use_ocr and load is not None and ocr_weight > 0.0
        self.ocr_weight = ocr_weight

        self.pixel_loss = nn.L1Loss()
        self.vgg_loss = VGG19PerceptualLoss() if self.use_perceptual else None
        self.ocr_loss = OCRLoss(load) if self.use_ocr else None

    def forward(self, sr, hr, lp_gt=None, loss_adv=None):
        pixel = self.pixel_loss(sr, hr)
        perceptual = (
            self.vgg_loss(sr, hr)
            if self.vgg_loss is not None
            else torch.zeros((), device=sr.device)
        )
        adversarial = (
            loss_adv if loss_adv is not None else torch.zeros((), device=sr.device)
        )
        ocr = torch.zeros((), device=sr.device)
        preds = None

        total = self.pixel_weight * pixel
        if self.vgg_loss is not None:
            total = total + self.perceptual_weight * perceptual
        if loss_adv is not None:
            total = total + self.adversarial_weight * adversarial
        if self.ocr_loss is not None and lp_gt is not None:
            ocr, preds = self.ocr_loss(sr, hr, lp_gt)
            total = total + self.ocr_weight * ocr

        metrics = {
            'pixel': pixel.detach().item(),
            'perceptual': perceptual.detach().item()
            if torch.is_tensor(perceptual)
            else float(perceptual),
            'adversarial': adversarial.detach().item()
            if torch.is_tensor(adversarial)
            else float(adversarial),
            'ocr': ocr.detach().item() if torch.is_tensor(ocr) else float(ocr),
            'total': total.detach().item(),
        }
        return total, metrics, preds


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub.set_dir(os.environ.get('TORCH_HOME', '/tmp/torch_cache'))
        try:
            if VGG19_Weights is not None:
                self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
            else:
                raise RuntimeError('VGG19_Weights unavailable')
        except Exception:
            try:
                self.vgg = vgg19(pretrained=True).features.eval()
            except Exception:
                self.vgg = vgg19(weights=None).features.eval()

        self.block_indices = [3, 8, 17, 26, 35]
        blocks = []
        last = 0
        for actual in self.block_indices:
            blocks.append(self.vgg[last:actual + 1])
            last = actual + 1
        self.blocks = nn.ModuleList(blocks)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = (x - 0.45) / 0.225
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class VGG19PerceptualLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.vgg = VGG19FeatureExtractor()
        self.l1 = nn.L1Loss()
        self.weights = weights or [0.1, 0.1, 1.0, 1.0, 1.0]

    def forward(self, sr, hr):
        self.vgg = self.vgg.to(sr.device)
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)

        loss = torch.zeros((), device=sr.device)
        for idx, (sr_feat, hr_feat) in enumerate(zip(sr_features, hr_features)):
            loss = loss + self.weights[idx] * self.l1(sr_feat, hr_feat)
        return loss


class OCRLoss(nn.Module):
    def __init__(self, load=None):
        super().__init__()
        self.load = load
        self.padding = False
        self.OCR = None

        if not load:
            return

        import tensorflow as tf
        from tensorflow.keras.utils import img_to_array

        self._tf = tf
        self._img_to_array = img_to_array

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self.load = Path(load)
        self.OCR = load_model(self.load.as_posix())
        self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
        self.parameters = np.load(
            self.load.as_posix() + '/parameters.npy', allow_pickle=True
        ).item()
        self.tasks = self.parameters['tasks']
        self.ocr_classes = self.parameters['ocr_classes']
        self.aspect_ratio = self.IMAGE_DIMS[1] / self.IMAGE_DIMS[0]
        self.min_ratio = self.aspect_ratio - 0.15
        self.max_ratio = self.aspect_ratio + 0.15
        self.padding = True
        self.OCR.compile()

    def OCR_pred(self, img, convert_to_bgr=True):
        if self.OCR is None:
            raise RuntimeError('OCRLoss was initialized without a valid OCR model')

        preds = []
        imgs = []

        if self.padding and convert_to_bgr:
            for im in img:
                im = np.array(im.detach().cpu().permute(1, 2, 0) * 255).astype('uint8')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                im, _, _ = padding(
                    im, self.min_ratio, self.max_ratio, color=(127, 127, 127)
                )
                imgs.append(im)

        for im in imgs:
            im = cv2.resize(im, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))
            im = self._img_to_array(im)
            im = im.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
            im = (im / 255.0).astype('float')
            predictions = self.OCR(im, training=False)

            plates = [''] * 1
            for task_idx, pp in enumerate(predictions):
                idx_aux = task_idx + 1
                task = self.tasks[task_idx]

                if re.match(r'^char[1-9]$', task):
                    for aux, pred in enumerate(pp):
                        plates[aux] += self.ocr_classes[f'char{idx_aux}'][np.argmax(pred)]
                else:
                    raise Exception(f"unknown task '{task}'!")
            preds.extend(plates)
        return preds

    def forward(self, sr_image, hr_image, label_texts):
        batch_size = sr_image.size(0)
        total_loss = torch.zeros((), device=sr_image.device)
        predicted_text = self.OCR_pred(sr_image)

        for idx in range(batch_size):
            if predicted_text[idx] != label_texts[idx]:
                total_loss = total_loss + torch.tensor(1.0, device=sr_image.device)

        return total_loss / batch_size, predicted_text
