from pathlib import Path

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
