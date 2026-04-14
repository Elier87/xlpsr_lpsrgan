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


def _sample_projection_offsets(horizontal_angle_deg, vertical_angle_deg):
    max_ratio_x = float(np.clip(np.tan(np.deg2rad(max(horizontal_angle_deg, 0.0))), 0.0, 0.55))
    max_ratio_y = float(np.clip(np.tan(np.deg2rad(max(vertical_angle_deg, 0.0))), 0.0, 0.35))
    if max_ratio_x <= 0 and max_ratio_y <= 0:
        return None

    left_inset = 0.0
    right_inset = 0.0
    top_inset = 0.0
    bottom_inset = 0.0

    if max_ratio_x > 0:
        horizontal_mag = random.uniform(0.0, max_ratio_x)
        if random.random() < 0.5:
            left_inset = horizontal_mag
        else:
            right_inset = horizontal_mag

    if max_ratio_y > 0:
        vertical_mag = random.uniform(0.0, max_ratio_y)
        if random.random() < 0.5:
            top_inset = vertical_mag
        else:
            bottom_inset = vertical_mag

    return np.array(
        [
            [left_inset, top_inset],
            [1.0 - right_inset, top_inset],
            [1.0 - right_inset, 1.0 - bottom_inset],
            [left_inset, 1.0 - bottom_inset],
        ],
        dtype=np.float32,
    )


def _apply_projection_transform(img, mask, normalized_dst, fill_color):
    if normalized_dst is None:
        return img, mask

    height, width = img.shape[:2]
    if height <= 1 or width <= 1:
        return img, mask

    src = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    dst = normalized_dst * scale

    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped_img = cv2.warpPerspective(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_img, warped_mask


class _BaseLPWrapper(Dataset):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        scale=2,
        projection_prob=0.0,
        projection_horizontal_angle=0.0,
        projection_vertical_angle=0.0,
        dataset=None,
    ):
        self.imgW = imgW
        self.imgH = imgH
        self.scale = scale
        self.aug = aug
        self.ar = image_aspect_ratio
        self.background = eval(background) if isinstance(background, str) else background
        self.projection_prob = float(projection_prob)
        self.projection_horizontal_angle = float(projection_horizontal_angle)
        self.projection_vertical_angle = float(projection_vertical_angle)
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

    def _prepare_image(self, img, size=None):
        if size is None:
            size = (self.imgH, self.imgW)
        img, _, _, _ = pad_with_mask(
            img, self.ar - 0.15, self.ar + 0.15, color=self.background
        )
        return resize_fn(img, size)


@register('lpsrgan')
class SRPairedImageWrapper(_BaseLPWrapper):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        scale=2,
        projection_prob=0.0,
        projection_horizontal_angle=0.0,
        projection_vertical_angle=0.0,
        rectify=False,
        dataset=None,
    ):
        super().__init__(
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            scale=scale,
            projection_prob=projection_prob,
            projection_horizontal_angle=projection_horizontal_angle,
            projection_vertical_angle=projection_vertical_angle,
            dataset=dataset,
        )
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

        if self.projection_prob > 0 and random.random() < self.projection_prob:
            normalized_dst = _sample_projection_offsets(
                self.projection_horizontal_angle,
                self.projection_vertical_angle,
            )
            lr_padded, lr_mask = _apply_projection_transform(
                lr_padded, lr_mask, normalized_dst, (127, 127, 127)
            )
            hr_padded, hr_mask = _apply_projection_transform(
                hr_padded, hr_mask, normalized_dst, (127, 127, 127)
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

            batch_lrs.append(self._prepare_image(img_lr, size=(self.imgH, self.imgW)))
            batch_hrs.append(
                self._prepare_image(
                    img_hr, size=(self.imgH * self.scale, self.imgW * self.scale)
                )
            )
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


def crop_bbox(img, bbox, margin_ratio=0.08):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2]


def frame_quality_score(img):
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(lap_var)


@register('challenge_sequence_wrapper')
class ChallengeSequenceWrapper(_BaseLPWrapper):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        mode='single',
        num_frames=1,
        bbox_margin=0.08,
        selection='quality',
        dataset=None,
    ):
        super().__init__(imgW, imgH, aug, image_aspect_ratio, background, dataset=dataset)
        self.mode = mode
        self.num_frames = num_frames
        self.bbox_margin = bbox_margin
        self.selection = selection

        assert self.dataset is not None, f'Not a valid dataset {self.dataset}'

    def _select_candidates(self, item):
        candidates = []
        for det in item['detections']:
            frame_name = det.get('frame')
            bbox = det.get('license_plate_coordinates')
            if frame_name is None or bbox is None:
                continue

            frame_path = item['sequence_dir'] / frame_name
            if not frame_path.exists():
                continue

            image = open_image(frame_path)
            crop = crop_bbox(image, bbox, margin_ratio=self.bbox_margin)
            if crop.size == 0:
                continue

            candidates.append(
                {
                    'frame_path': frame_path,
                    'frame_name': frame_name,
                    'bbox': bbox,
                    'crop': crop,
                    'score': frame_quality_score(crop),
                }
            )

        if not candidates:
            for frame_path in item['frames']:
                image = open_image(frame_path)
                candidates.append(
                    {
                        'frame_path': frame_path,
                        'frame_name': frame_path.name,
                        'bbox': None,
                        'crop': image,
                        'score': frame_quality_score(image),
                    }
                )

        if self.selection == 'quality':
            candidates.sort(key=lambda x: x['score'], reverse=True)
        else:
            candidates.sort(key=lambda x: x['frame_name'])

        return candidates

    def collate_fn(self, datas):
        batch_lrs = []
        batch_gt = []
        batch_names = []
        batch_sequence_ids = []
        batch_frame_names = []
        batch_bboxes = []
        batch_scores = []

        for item in datas:
            candidates = self._select_candidates(item)
            take = 1 if self.mode == 'single' else max(1, self.num_frames)
            selected = candidates[:take]

            crops = [self._prepare_image(candidate['crop']) for candidate in selected]
            if self.mode == 'single':
                batch_lrs.append(crops[0])
            else:
                batch_lrs.append(torch.stack(crops))

            batch_gt.append(item.get('gt'))
            batch_names.append(item['sequence_id'])
            batch_sequence_ids.append(item['sequence_id'])
            batch_frame_names.append([candidate['frame_name'] for candidate in selected])
            batch_bboxes.append([candidate['bbox'] for candidate in selected])
            batch_scores.append([candidate['score'] for candidate in selected])

        return {
            'lr': torch.stack(batch_lrs),
            'gt': batch_gt,
            'name': batch_names,
            'sequence_id': batch_sequence_ids,
            'frame_names': batch_frame_names,
            'bboxes': batch_bboxes,
            'quality_scores': batch_scores,
        }


@register('challenge_finetune_wrapper')
class ChallengeFinetuneWrapper(ChallengeSequenceWrapper):
    def __init__(
        self,
        imgW,
        imgH,
        aug,
        image_aspect_ratio,
        background,
        bbox_margin=0.08,
        selection='quality',
        topk_pool=5,
        sample_strategy='random_topk',
        dataset=None,
    ):
        super().__init__(
            imgW=imgW,
            imgH=imgH,
            aug=aug,
            image_aspect_ratio=image_aspect_ratio,
            background=background,
            mode='single',
            num_frames=1,
            bbox_margin=bbox_margin,
            selection=selection,
            dataset=dataset,
        )
        self.topk_pool = max(1, int(topk_pool))
        self.sample_strategy = sample_strategy

    def _pick_candidate(self, candidates):
        if not candidates:
            raise RuntimeError('No challenge candidates available for finetune wrapper')

        if self.sample_strategy == 'best':
            return candidates[0]
        if self.sample_strategy == 'random_all':
            return random.choice(candidates)
        if self.sample_strategy == 'random_topk':
            return random.choice(candidates[:min(len(candidates), self.topk_pool)])

        raise ValueError(f'Unsupported challenge sample_strategy: {self.sample_strategy}')

    def collate_fn(self, datas):
        batch_lrs = []
        batch_gt = []
        batch_names = []
        batch_sequence_ids = []
        batch_frame_names = []
        batch_bboxes = []
        batch_scores = []

        for item in datas:
            candidates = self._select_candidates(item)
            candidate = self._pick_candidate(candidates)

            batch_lrs.append(self._prepare_image(candidate['crop']))
            batch_gt.append(item.get('gt'))
            batch_names.append(item['sequence_id'])
            batch_sequence_ids.append(item['sequence_id'])
            batch_frame_names.append(candidate['frame_name'])
            batch_bboxes.append(candidate['bbox'])
            batch_scores.append(candidate['score'])

        return {
            'lr': torch.stack(batch_lrs),
            'gt': batch_gt,
            'name': batch_names,
            'sequence_id': batch_sequence_ids,
            'frame_name': batch_frame_names,
            'bbox': batch_bboxes,
            'quality_score': batch_scores,
        }
