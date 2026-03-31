from collections import Counter

import torch


def normalize_text(text):
    return ''.join(ch for ch in (text or '').upper() if ch.isalnum())


def majority_vote_by_position(texts):
    texts = [normalize_text(text) for text in texts if text is not None]
    if not texts:
        return ''

    max_len = max(len(text) for text in texts)
    voted = []
    for idx in range(max_len):
        chars = [text[idx] for text in texts if idx < len(text)]
        if not chars:
            continue
        voted.append(Counter(chars).most_common(1)[0][0])
    return ''.join(voted)


def select_highest_confidence(texts, confidences):
    if not texts:
        return ''
    if not isinstance(confidences, torch.Tensor):
        confidences = torch.tensor(confidences, dtype=torch.float32)
    best_idx = int(torch.argmax(confidences).item())
    return normalize_text(texts[best_idx])


def select_most_frequent(texts, confidences):
    texts = [normalize_text(text) for text in texts]
    if not texts:
        return ''

    counts = Counter(texts)
    max_count = max(counts.values())
    winners = {text for text, count in counts.items() if count == max_count}
    if len(winners) == 1:
        return next(iter(winners))

    if not isinstance(confidences, torch.Tensor):
        confidences = torch.tensor(confidences, dtype=torch.float32)
    best_idx = max(
        (idx for idx, text in enumerate(texts) if text in winners),
        key=lambda idx: float(confidences[idx]),
    )
    return texts[best_idx]


def aggregate_sequence_predictions(texts, confidences, strategy='most_frequent'):
    if strategy == 'highest_confidence':
        return select_highest_confidence(texts, confidences)
    if strategy == 'character_vote':
        return majority_vote_by_position(texts)
    return select_most_frequent(texts, confidences)
