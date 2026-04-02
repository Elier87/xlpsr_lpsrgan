import csv
import json
from pathlib import Path

from torch.utils.data import Dataset

from datasets import register


def _load_ground_truth_map(path):
    if path is None:
        return {}

    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return {
            row['folder'].strip(): row['license_plate'].strip()
            for row in reader
            if row.get('folder')
        }


def _slice_by_phase(seq_dirs, phase=None, train_ratio=0.8, val_ratio=0.2):
    if phase is None or str(phase).lower() in ('all', 'full'):
        return seq_dirs

    phase = str(phase).lower()
    total = len(seq_dirs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if phase == 'training':
        return seq_dirs[:train_end]
    if phase == 'validation':
        return seq_dirs[train_end:val_end]
    if phase == 'testing':
        return seq_dirs[val_end:]

    raise ValueError(f'Unsupported challenge phase: {phase}')


@register('challenge_sequence')
class ChallengeSequenceDataset(Dataset):
    def __init__(
        self,
        root,
        ground_truth_csv=None,
        sequence_glob='seq_*',
        max_sequences=None,
        include_empty=False,
        phase=None,
        train_ratio=0.8,
        val_ratio=0.2,
    ):
        self.root = Path(root)
        self.gt_map = _load_ground_truth_map(ground_truth_csv)
        self.dataset = []

        if not self.root.exists():
            raise FileNotFoundError(f'Challenge root not found: {self.root}')

        seq_dirs = sorted(
            path for path in self.root.glob(sequence_glob) if path.is_dir()
        )
        seq_dirs = _slice_by_phase(
            seq_dirs, phase=phase, train_ratio=train_ratio, val_ratio=val_ratio
        )
        if max_sequences is not None:
            seq_dirs = seq_dirs[:max_sequences]

        for seq_dir in seq_dirs:
            detections_path = seq_dir / 'detections.json'
            if not detections_path.exists():
                if include_empty:
                    detections = []
                else:
                    continue
            else:
                detections = json.loads(detections_path.read_text())

            frames = sorted(seq_dir.glob('*.png'))
            if not frames and not include_empty:
                continue

            self.dataset.append(
                {
                    'sequence_id': seq_dir.name,
                    'sequence_dir': seq_dir,
                    'frames': frames,
                    'detections': detections,
                    'gt': self.gt_map.get(seq_dir.name),
                }
            )

        if not self.dataset:
            raise RuntimeError(f'No challenge sequences found under {self.root}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
