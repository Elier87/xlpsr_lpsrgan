import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)
        )
    )


def open_image(img_path, cvt=True):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Unable to read image: {img_path}')
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pad_with_mask(img, min_ratio, max_ratio, color=(0, 0, 0)):
    img_h, img_w = np.shape(img)[:2]
    border_w = 0
    border_h = 0
    ar = float(img_w) / img_h
    mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

    if min_ratio <= ar <= max_ratio:
        mask = np.zeros_like(img, dtype=np.uint8)
        return img, mask, border_w, border_h

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
    mask = cv2.copyMakeBorder(
        mask, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=0
    )

    return img, mask, border_w, border_h


def crop_to_valid_region(img, mask):
    valid_pixels = np.where(mask == 1)
    if len(valid_pixels[0]) == 0:
        return img, mask

    min_y, max_y = np.min(valid_pixels[0]), np.max(valid_pixels[0])
    min_x, max_x = np.min(valid_pixels[1]), np.max(valid_pixels[1])

    return (
        img[min_y:max_y + 1, min_x:max_x + 1],
        mask[min_y:max_y + 1, min_x:max_x + 1],
    )


def rectify_img(img, pts, margin=2):
    (tl, tr, br, bl) = pts

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    max_width += margin * 2
    max_height += margin * 2

    ww = max_width - 1 - margin
    hh = max_height - 1 - margin
    dst = np.array(
        [[margin, margin], [ww, margin], [ww, hh], [margin, hh]], dtype='float32'
    )

    pts = np.array(pts, dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (max_width, max_height))


def get_pts(file_path):
    with open(file_path.with_suffix('.json'), 'r') as j:
        return json.load(j)['shapes'][0]['points']


class _BaseLPWrapper(Dataset):
    def __init__(self, imgW, imgH, aug, image_aspect_ratio, background, dataset=None):
        self.imgW = imgW
        self.imgH = imgH
        self.aug = aug
        self.ar = image_aspect_ratio
        self.background = eval(background) if isinstance(background, str) else background
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _make_transform(self):
        transform = A.OneOf(
            [
                A.SafeRotate(limit=2, border_mode=cv2.BORDER_REPLICATE, p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.2, p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.NoOp(p=0.1),
            ],
            p=1.0,
        )
        return A.Compose(
            [transform],
            additional_targets={'image2': 'image', 'mask2': 'mask'},
            is_check_shapes=False,
        )

    def _prepare_image(self, img):
        img, _, _, _ = pad_with_mask(
            img, self.ar - 0.15, self.ar + 0.15, color=self.background
        )
        return resize_fn(img, (self.imgH, self.imgW))


@register('lpsrgan')
class SRPairedImageWrapper(_BaseLPWrapper):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        rectify=False,
        dataset=None,
    ):
        super().__init__(imgW, imgH, aug, image_aspect_ratio, background, dataset=dataset)
        self.rectify = rectify
        self.transform = self._make_transform()

        assert self.dataset is not None, f'Not a valid dataset {self.dataset}'

    def _augment_pair(self, lr_img, hr_img):
        lr_padded, lr_mask, _, _ = pad_with_mask(
            lr_img, self.ar - 0.15, self.ar + 0.15, color=(127, 127, 127)
        )
        hr_padded, hr_mask, _, _ = pad_with_mask(
            hr_img, self.ar - 0.15, self.ar + 0.15, color=(127, 127, 127)
        )

        augmented = self.transform(
            image=lr_padded, image2=hr_padded, mask=lr_mask, mask2=hr_mask
        )
        lr_aug, _ = crop_to_valid_region(augmented['image'], augmented['mask'])
        hr_aug, _ = crop_to_valid_region(augmented['image2'], augmented['mask2'])
        return lr_aug, hr_aug

    def collate_fn(self, datas):
        batch_lrs = []
        batch_hrs = []
        batch_plates = []
        batch_names = []

        for item in datas:
            if 'lr_path' not in item or 'hr_path' not in item:
                raise KeyError(
                    'Stage 1 paired training expects dataset samples with lr_path and hr_path'
                )

            lr_path = Path(item['lr_path'])
            hr_path = Path(item['hr_path'])
            img_lr = open_image(lr_path)
            img_hr = open_image(hr_path)

            if self.rectify and lr_path.with_suffix('.json').exists():
                img_lr = rectify_img(img_lr, get_pts(lr_path), margin=2)
            if self.rectify and hr_path.with_suffix('.json').exists():
                img_hr = rectify_img(img_hr, get_pts(hr_path), margin=2)

            if self.aug:
                img_lr, img_hr = self._augment_pair(img_lr, img_hr)

            batch_lrs.append(self._prepare_image(img_lr))
            batch_hrs.append(self._prepare_image(img_hr))
            batch_plates.append(item['gt'])
            batch_names.append(item.get('name', lr_path.stem))

        return {
            'lr': torch.stack(batch_lrs),
            'hr': torch.stack(batch_hrs),
            'gt': batch_plates,
            'name': batch_names,
        }


@register('SR_multi_image')
class SR_multi_image(_BaseLPWrapper):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        test=False,
        in_images=5,
        time_series=False,
        rectify=False,
        dataset=None,
    ):
        super().__init__(imgW, imgH, aug, image_aspect_ratio, background, dataset=dataset)
        self.test = test
        self.in_images = in_images
        self.time_series = time_series
        self.rectify = rectify

        assert self.dataset is not None, f'Not a valid dataset {self.dataset}'

    def collate_fn(self, datas):
        batch_lrs = []
        batch_plates = []
        file_name = []

        for item in datas:
            lr_imgs = []
            paths = sorted(list(item['imgs'].glob('*.png')))

            if self.test:
                paths = paths[:self.in_images]
            else:
                paths = [random.choice(paths)]

            path_lp = next(item['imgs'].rglob('plate-*.txt'))
            plate = path_lp.read_text().splitlines()[0].strip()

            batch_plates.append(plate)
            file_name.append(path_lp)

            for lr_path in paths:
                img_lr = open_image(lr_path)

                if self.rectify and lr_path.with_suffix('.json').exists():
                    img_lr = rectify_img(img_lr, get_pts(lr_path), margin=2)

                lr_imgs.append(self._prepare_image(img_lr))

            batch_lrs.append(
                torch.cat(lr_imgs, dim=0) if not self.time_series else torch.stack(lr_imgs)
            )

        return {
            'lr': torch.stack(batch_lrs),
            'gt': batch_plates,
            'name': file_name if self.test else None,
        }
