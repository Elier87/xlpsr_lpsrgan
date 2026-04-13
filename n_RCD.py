import argparse
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping')
    return config


def set_seed(seed):
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _is_lr_png(path):
    return path.is_file() and path.suffix.lower() == '.png' and path.name.lower().startswith('lr-')


def copy_dataset_tree(input_root, output_root, copy_non_lr_files=True, overwrite=False):
    input_root = Path(input_root)
    output_root = Path(output_root)

    if input_root.resolve() == output_root.resolve():
        raise ValueError('input_root and output_root must be different')

    output_root.mkdir(parents=True, exist_ok=True)
    copied_files = 0

    for src_path in sorted(input_root.rglob('*')):
        rel_path = src_path.relative_to(input_root)
        dst_path = output_root / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        if not copy_non_lr_files:
            continue

        if _is_lr_png(src_path):
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and not overwrite:
            continue
        shutil.copy2(src_path, dst_path)
        copied_files += 1

    print(f'[copy_dataset_tree] Copied {copied_files} non-LR files into {output_root}')


def _read_image_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path, image, save_dtype='uint8', save_format='png'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_dtype != 'uint8':
        raise ValueError(f'Unsupported save_dtype: {save_dtype}')
    if str(save_format).lower() != 'png':
        raise ValueError(f'Unsupported save_format: {save_format}. This utility only writes PNG.')

    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise IOError(f'Failed to save image: {path}')


def _normalize_image(image):
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _choice_from_prob_dict(prob_dict, rng):
    keys = list(prob_dict.keys())
    probs = np.array([float(prob_dict[key]) for key in keys], dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError(f'Negative probability found: {prob_dict}')
    prob_sum = probs.sum()
    if prob_sum <= 0:
        raise ValueError(f'Probability sum must be > 0: {prob_dict}')
    probs = probs / prob_sum
    idx = int(rng.choice(len(keys), p=probs))
    return keys[idx]


def _sample_odd_kernel_size(kernel_sizes, rng):
    valid = [int(k) for k in kernel_sizes if int(k) > 1 and int(k) % 2 == 1]
    if not valid:
        raise ValueError(f'No valid odd kernel sizes found: {kernel_sizes}')
    return int(rng.choice(valid))


def _sample_odd_from_range(value_range, rng):
    low, high = int(value_range[0]), int(value_range[1])
    if low > high:
        low, high = high, low
    candidates = [k for k in range(low, high + 1) if k % 2 == 1 and k > 1]
    if not candidates:
        raise ValueError(f'No valid odd numbers in range: {value_range}')
    return int(rng.choice(candidates))


def _apply_kernel(image, kernel):
    image = _normalize_image(image)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _build_gaussian_kernel(kernel_size, sigma):
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def _build_generalized_gaussian_kernel(kernel_size, sigma, beta):
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.exp(-np.power(np.maximum(rr / max(sigma, 1e-6), 0.0), beta))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def _build_plateau_kernel(kernel_size, sigma, beta):
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = 1.0 / (1.0 + np.power(np.maximum(rr / max(sigma, 1e-6), 0.0), beta))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def apply_blur(image, blur_cfg, rng):
    image = _normalize_image(image)
    kernel_type = _choice_from_prob_dict(blur_cfg['kernel_type_prob'], rng)
    kernel_size = _sample_odd_kernel_size(blur_cfg['kernel_sizes'], rng)
    sigma = float(rng.uniform(float(blur_cfg['sigma_range'][0]), float(blur_cfg['sigma_range'][1])))

    if kernel_type == 'gaussian':
        kernel = _build_gaussian_kernel(kernel_size, sigma)
    elif kernel_type == 'generalized_gaussian':
        beta = float(
            rng.uniform(
                float(blur_cfg['generalized_beta_range'][0]),
                float(blur_cfg['generalized_beta_range'][1]),
            )
        )
        kernel = _build_generalized_gaussian_kernel(kernel_size, sigma, beta)
    elif kernel_type == 'plateau':
        beta = float(
            rng.uniform(
                float(blur_cfg['plateau_beta_range'][0]),
                float(blur_cfg['plateau_beta_range'][1]),
            )
        )
        kernel = _build_plateau_kernel(kernel_size, sigma, beta)
    else:
        raise ValueError(f'Unsupported blur kernel type: {kernel_type}')

    return _apply_kernel(image, kernel)


def _get_interpolation_code(name):
    name = str(name).lower()
    mapping = {
        'area': cv2.INTER_AREA,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    if name not in mapping:
        raise ValueError(f'Unsupported interpolation mode: {name}')
    return mapping[name]


def _choose_interpolation(resize_cfg, rng):
    pool = [str(name).lower() for name in resize_cfg['interpolation_pool']]
    if not resize_cfg.get('use_nearest', False):
        pool = [name for name in pool if name != 'nearest']
    if not pool:
        raise ValueError('Interpolation pool is empty after filtering nearest')
    return _get_interpolation_code(rng.choice(pool))


def apply_resize_stage(image, resize_cfg, rng):
    image = _normalize_image(image)
    height, width = image.shape[:2]
    mode = _choice_from_prob_dict(resize_cfg['mode_prob'], rng)
    scale_min = float(resize_cfg['scale_range'][0])
    scale_max = float(resize_cfg['scale_range'][1])
    if scale_min > scale_max:
        scale_min, scale_max = scale_max, scale_min

    if mode == 'upsample':
        low = max(1.0, scale_min)
        high = max(low, scale_max)
        scale = float(rng.uniform(low, high))
    elif mode == 'downsample':
        low = min(scale_min, 1.0)
        high = min(scale_max, 1.0)
        if high <= 0:
            high = 1.0
        scale = float(rng.uniform(low, high))
    elif mode == 'keep':
        scale = 1.0
    else:
        raise ValueError(f'Unsupported resize mode: {mode}')

    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    interpolation = _choose_interpolation(resize_cfg, rng)
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def apply_gaussian_noise(image, noise_cfg, rng):
    image = _normalize_image(image)
    std = float(
        rng.uniform(
            float(noise_cfg['gaussian_std_range'][0]),
            float(noise_cfg['gaussian_std_range'][1]),
        )
    ) / 255.0
    grayscale_noise = rng.random() < float(noise_cfg['grayscale_noise_prob'])

    if grayscale_noise:
        noise = rng.normal(0.0, std, size=(image.shape[0], image.shape[1], 1)).astype(np.float32)
    else:
        noise = rng.normal(0.0, std, size=image.shape).astype(np.float32)

    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


def apply_poisson_noise(image, noise_cfg, rng):
    image = _normalize_image(image)
    scale = float(
        rng.uniform(
            float(noise_cfg['poisson_scale_range'][0]),
            float(noise_cfg['poisson_scale_range'][1]),
        )
    )
    grayscale_noise = rng.random() < float(noise_cfg['grayscale_noise_prob'])

    if grayscale_noise:
        base = np.mean(image, axis=2, keepdims=True)
        lam = np.clip(base, 0.0, 1.0) * 255.0 * scale
        noisy_base = rng.poisson(lam).astype(np.float32) / (255.0 * scale)
        delta = noisy_base - base
        noisy = image + delta
    else:
        lam = np.clip(image, 0.0, 1.0) * 255.0 * scale
        noisy = rng.poisson(lam).astype(np.float32) / (255.0 * scale)

    return np.clip(noisy, 0.0, 1.0)


def apply_jpeg_compression(image, jpeg_cfg, rng):
    image = _normalize_image(image)
    quality = int(
        round(
            rng.uniform(
                float(jpeg_cfg['quality_range'][0]),
                float(jpeg_cfg['quality_range'][1]),
            )
        )
    )
    quality = int(np.clip(quality, 0, 100))

    image_uint8 = (image * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError('JPEG encoding failed during degradation')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise IOError('JPEG decoding failed during degradation')
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(decoded, 0.0, 1.0)


def _build_sinc_kernel(kernel_size, cutoff):
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    window = np.hamming(kernel_size).astype(np.float32)
    sinc_1d = np.sinc(cutoff * coords).astype(np.float32) * window
    kernel = np.outer(sinc_1d, sinc_1d)
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def apply_sinc_ringing(image, ringing_cfg, rng):
    image = _normalize_image(image)
    apply_prob = float(ringing_cfg.get('apply_prob', 0.0))
    if rng.random() >= apply_prob:
        return image

    kernel_size = _sample_odd_from_range(ringing_cfg['kernel_size_range'], rng)
    cutoff = float(
        rng.uniform(
            float(ringing_cfg['cutoff_range'][0]),
            float(ringing_cfg['cutoff_range'][1]),
        )
    )
    kernel = _build_sinc_kernel(kernel_size, cutoff)
    return _apply_kernel(image, kernel)


def degrade_one_stage(image, nrcd_cfg, rng):
    operations = [
        lambda img: apply_blur(img, nrcd_cfg['blur'], rng),
        lambda img: apply_resize_stage(img, nrcd_cfg['resize'], rng),
        lambda img: apply_jpeg_compression(img, nrcd_cfg['jpeg'], rng),
        lambda img: apply_sinc_ringing(img, nrcd_cfg['ringing'], rng),
    ]

    if rng.random() < float(nrcd_cfg['noise']['gaussian_prob']):
        operations.append(lambda img: apply_gaussian_noise(img, nrcd_cfg['noise'], rng))
    if rng.random() < float(nrcd_cfg['noise']['poisson_prob']):
        operations.append(lambda img: apply_poisson_noise(img, nrcd_cfg['noise'], rng))

    rng.shuffle(operations)
    degraded = _normalize_image(image)
    for operation in operations:
        degraded = operation(degraded)
    return np.clip(degraded, 0.0, 1.0)


def degrade_n_stage(image, nrcd_cfg, target_size, rng):
    image = _normalize_image(image)
    n_stage = int(nrcd_cfg['n_stage'])
    stage_apply_prob = list(nrcd_cfg.get('stage_apply_prob', []))
    if not stage_apply_prob:
        stage_apply_prob = [1.0] * n_stage
    if len(stage_apply_prob) < n_stage:
        stage_apply_prob.extend([stage_apply_prob[-1]] * (n_stage - len(stage_apply_prob)))

    degraded = image
    for stage_idx in range(n_stage):
        apply_prob = float(stage_apply_prob[stage_idx])
        if rng.random() <= apply_prob:
            degraded = degrade_one_stage(degraded, nrcd_cfg, rng)

    final_interpolation = _choose_interpolation(nrcd_cfg['resize'], rng)
    degraded = cv2.resize(
        degraded,
        (int(target_size[0]), int(target_size[1])),
        interpolation=final_interpolation,
    )
    return np.clip(degraded, 0.0, 1.0)


def find_hr_lr_pairs(plate_dir):
    plate_dir = Path(plate_dir)
    hr_paths = sorted(plate_dir.glob('hr-*.png'))
    pairs = []
    for hr_path in hr_paths:
        lr_name = hr_path.name.replace('hr-', 'lr-', 1)
        lr_path = plate_dir / lr_name
        pairs.append((hr_path, lr_path if lr_path.exists() else None))
    return pairs


def _derive_target_size(hr_path, existing_lr_path, matching_cfg):
    if bool(matching_cfg.get('match_existing_lr_size', True)) and existing_lr_path is not None:
        lr_image = cv2.imread(str(existing_lr_path), cv2.IMREAD_COLOR)
        if lr_image is not None:
            height, width = lr_image.shape[:2]
            return int(width), int(height)
        warnings.warn(f'Unable to read existing LR image size from {existing_lr_path}, falling back')

    hr_image = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    if hr_image is None:
        raise ValueError(f'Failed to read HR image for fallback sizing: {hr_path}')
    hr_height, hr_width = hr_image.shape[:2]
    scale_factor = float(matching_cfg['fallback_scale_factor'])
    width = max(1, int(round(hr_width / scale_factor)))
    height = max(1, int(round(hr_height / scale_factor)))
    return width, height


def process_plate_dir(plate_dir, output_plate_dir, config, rng):
    plate_dir = Path(plate_dir)
    output_plate_dir = Path(output_plate_dir)
    output_plate_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_hr_lr_pairs(plate_dir)
    if not pairs:
        warnings.warn(f'No HR files found in {plate_dir}, skipping')
        return {'processed': 0, 'skipped': 0}

    overwrite = bool(config['dataset'].get('overwrite', False))
    processed = 0
    skipped = 0

    for hr_path, existing_lr_path in pairs:
        output_lr_name = hr_path.name.replace('hr-', 'lr-', 1)
        output_lr_path = output_plate_dir / output_lr_name

        if output_lr_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            hr_image = _normalize_image(_read_image_rgb(hr_path))
            target_size = _derive_target_size(hr_path, existing_lr_path, config['matching'])
            degraded = degrade_n_stage(hr_image, config['nrcd'], target_size, rng)
            _write_image_rgb(
                output_lr_path,
                degraded,
                save_dtype=config['output']['save_dtype'],
                save_format=config['output']['save_format'],
            )
            processed += 1
        except Exception as exc:
            warnings.warn(f'Failed to process {hr_path}: {exc}')
            skipped += 1

    return {'processed': processed, 'skipped': skipped}


def process_dataset(config):
    input_root = Path(config['dataset']['input_root'])
    output_root = Path(config['dataset']['output_root'])

    if not input_root.exists():
        raise FileNotFoundError(f'Input root not found: {input_root}')

    rng = set_seed(int(config['dataset'].get('seed', 123)))
    copy_dataset_tree(
        input_root=input_root,
        output_root=output_root,
        copy_non_lr_files=bool(config['dataset'].get('copy_non_lr_files', True)),
        overwrite=bool(config['dataset'].get('overwrite', False)),
    )

    plate_dirs = []
    for directory in sorted(path for path in input_root.rglob('*') if path.is_dir()):
        if any(directory.glob('hr-*.png')):
            plate_dirs.append(directory)

    if not plate_dirs:
        warnings.warn(f'No plate directories with HR images found under {input_root}')
        return

    total_processed = 0
    total_skipped = 0
    total_dirs = len(plate_dirs)

    for idx, plate_dir in enumerate(plate_dirs, start=1):
        rel_dir = plate_dir.relative_to(input_root)
        output_plate_dir = output_root / rel_dir
        stats = process_plate_dir(plate_dir, output_plate_dir, config, rng)
        total_processed += stats['processed']
        total_skipped += stats['skipped']
        print(
            f'[process_dataset] {idx}/{total_dirs} {rel_dir} '
            f'processed={stats["processed"]} skipped={stats["skipped"]}'
        )

    print(
        f'[process_dataset] Done. total_processed={total_processed} '
        f'total_skipped={total_skipped} output_root={output_root}'
    )


def main():
    parser = argparse.ArgumentParser(description='Offline UFPR n-stage random combination degradation')
    parser.add_argument('--config', required=True, help='Path to n_RCD.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    process_dataset(config)


if __name__ == '__main__':
    main()
