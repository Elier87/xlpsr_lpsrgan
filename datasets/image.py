from pathlib import Path
import json

from torch.utils.data import Dataset

from datasets import register


def _read_split_file(path_split):
    with open(path_split, 'r') as f:
        return [line.strip() for line in f if line.strip()]


@register('multi_image')
class multi_image(Dataset):
    def __init__(self, path_split, phase='training'):
        self.split_file = path_split
        self.phase = phase
        self.dataset = []

        for line in _read_split_file(self.split_file):
            gt, path_imgs, split = line.split(';')
            sample = {
                'gt': gt.strip(),
                'imgs': Path(path_imgs.strip()),
            }

            if self.phase in split:
                self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


@register('paired_image')
class paired_image(Dataset):
    def __init__(
        self,
        path_split,
        phase='training',
        lr_pattern='lr-*.png',
        hr_pattern='hr-*.png',
        plate_glob='plate-*.txt',
    ):
        self.split_file = path_split
        self.phase = phase
        self.lr_pattern = lr_pattern
        self.hr_pattern = hr_pattern
        self.plate_glob = plate_glob
        self.dataset = []

        for line in _read_split_file(self.split_file):
            gt, path_imgs, split = line.split(';')
            if self.phase not in split:
                continue

            folder = Path(path_imgs.strip())
            gt = gt.strip()

            if not folder.exists():
                raise FileNotFoundError(f'Paired dataset folder not found: {folder}')

            if not gt:
                gt_files = sorted(folder.glob(self.plate_glob))
                if not gt_files:
                    raise FileNotFoundError(f'No plate label file found under: {folder}')
                gt = gt_files[0].read_text().strip()

            lr_paths = sorted(folder.glob(self.lr_pattern))
            if not lr_paths:
                raise FileNotFoundError(f'No LR images found under: {folder}')

            for lr_path in lr_paths:
                hr_name = lr_path.name.replace('lr-', 'hr-', 1)
                hr_path = lr_path.with_name(hr_name)

                if not hr_path.exists():
                    raise FileNotFoundError(
                        f'Missing HR pair for {lr_path.name} in {folder}'
                    )

                self.dataset.append(
                    {
                        'gt': gt,
                        'name': f'{folder.name}/{lr_path.stem}',
                        'folder': folder,
                        'lr_path': lr_path,
                        'hr_path': hr_path,
                    }
                )

        if not self.dataset:
            raise RuntimeError(
                f'No paired samples found for phase "{self.phase}" in split "{self.split_file}"'
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def _load_ufpr_plate(json_path):
    data = json.loads(Path(json_path).read_text())
    return data['plate'].strip()


@register('ufpr_paired')
class ufpr_paired(Dataset):
    def __init__(
        self,
        roots,
        phase='training',
        lr_pattern='lr-*.png',
        hr_pattern='hr-*.png',
        train_ratio=0.8,
        val_ratio=0.1,
        max_folders=None,
    ):
        self.phase = phase
        if isinstance(roots, str):
            roots = [roots]
        self.roots = [Path(root) for root in roots]
        self.dataset = []

        plate_folders = []
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f'UFPR root not found: {root}')
            plate_folders.extend(sorted(path for path in root.rglob('plate-*') if path.is_dir()))

        if not plate_folders:
            raise RuntimeError(f'No UFPR plate folders found under: {self.roots}')

        plate_folders = sorted(plate_folders)
        if max_folders is not None:
            plate_folders = plate_folders[:max_folders]

        total = len(plate_folders)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if phase == 'training':
            selected_folders = plate_folders[:train_end]
        elif phase == 'validation':
            selected_folders = plate_folders[train_end:val_end]
        elif phase == 'testing':
            selected_folders = plate_folders[val_end:]
        else:
            raise ValueError(f'Unknown phase: {phase}')

        if not selected_folders:
            raise RuntimeError(f'No UFPR folders selected for phase "{phase}"')

        for folder in selected_folders:
            lr_paths = sorted(folder.glob(lr_pattern))
            if not lr_paths:
                continue

            for lr_path in lr_paths:
                hr_name = lr_path.name.replace('lr-', 'hr-', 1)
                hr_path = lr_path.with_name(hr_name)
                if not hr_path.exists():
                    raise FileNotFoundError(f'Missing HR pair for {lr_path}')

                label_json = hr_path.with_suffix('.json')
                if not label_json.exists():
                    label_json = lr_path.with_suffix('.json')
                if not label_json.exists():
                    raise FileNotFoundError(f'Missing label json for {lr_path}')

                gt = _load_ufpr_plate(label_json)
                rel_parent = next(
                    (str(folder.relative_to(root)) for root in self.roots if folder.is_relative_to(root)),
                    folder.name,
                )
                self.dataset.append(
                    {
                        'gt': gt,
                        'name': f'{rel_parent}/{lr_path.stem}',
                        'folder': folder,
                        'lr_path': lr_path,
                        'hr_path': hr_path,
                    }
                )

        if not self.dataset:
            raise RuntimeError(f'No UFPR paired samples found for phase "{phase}"')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
