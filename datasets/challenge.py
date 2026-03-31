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


@register('challenge_sequence')
class ChallengeSequenceDataset(Dataset):
    def __init__(
        self,
        root,
        ground_truth_csv=None,
        sequence_glob='seq_*',
        max_sequences=None,
        include_empty=False,
    ):
        self.root = Path(root)
        self.gt_map = _load_ground_truth_map(ground_truth_csv)
        self.dataset = []

        if not self.root.exists():
            raise FileNotFoundError(f'Challenge root not found: {self.root}')

        seq_dirs = sorted(
            path for path in self.root.glob(sequence_glob) if path.is_dir()
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
