import torch

from .plate_format import normalize_plate_text


FRLP_BLANK_TOKEN = '[BLANK]'
FRLP_EOS_TOKEN = '[EOS]'
FRLP_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
FRLP_TOKENS = [FRLP_BLANK_TOKEN, FRLP_EOS_TOKEN] + list(FRLP_CHARSET)
FRLP_TOKEN_TO_ID = {token: idx for idx, token in enumerate(FRLP_TOKENS)}
FRLP_ID_TO_TOKEN = {idx: token for token, idx in FRLP_TOKEN_TO_ID.items()}

FRLP_BLANK_ID = FRLP_TOKEN_TO_ID[FRLP_BLANK_TOKEN]
FRLP_EOS_ID = FRLP_TOKEN_TO_ID[FRLP_EOS_TOKEN]
FRLP_VOCAB_SIZE = len(FRLP_TOKENS)

FRLP_TYPE_BLANK = 0
FRLP_TYPE_EOS = 1
FRLP_TYPE_LETTER = 2
FRLP_TYPE_DIGIT = 3
FRLP_TYPE_VOCAB_SIZE = 4


def char_to_type_id(char):
    if char.isalpha():
        return FRLP_TYPE_LETTER
    if char.isdigit():
        return FRLP_TYPE_DIGIT
    return FRLP_TYPE_BLANK


def score_to_unit_interval(score, min_score=-7.0, max_score=14.0):
    score = float(score)
    return (score - min_score) / (max_score - min_score)


def normalize_frlp_text(text):
    return normalize_plate_text(text)


def encode_frlp_targets(targets, max_steps=9):
    batch_size = len(targets)
    token_targets = torch.full((batch_size, max_steps), FRLP_BLANK_ID, dtype=torch.long)
    type_targets = torch.full((batch_size, max_steps), FRLP_TYPE_BLANK, dtype=torch.long)
    eos_positions = torch.zeros(batch_size, dtype=torch.long)
    valid_counts = torch.zeros(batch_size, dtype=torch.long)
    normalized_targets = []

    for batch_idx, target in enumerate(targets):
        normalized = normalize_frlp_text(target)
        normalized = normalized[:max(max_steps - 1, 1)]
        normalized_targets.append(normalized)

        valid_counts[batch_idx] = len(normalized)
        for step_idx, char in enumerate(normalized):
            token_targets[batch_idx, step_idx] = FRLP_TOKEN_TO_ID[char]
            type_targets[batch_idx, step_idx] = char_to_type_id(char)

        eos_idx = min(len(normalized), max_steps - 1)
        token_targets[batch_idx, eos_idx] = FRLP_EOS_ID
        type_targets[batch_idx, eos_idx] = FRLP_TYPE_EOS
        eos_positions[batch_idx] = eos_idx

    return {
        'token_targets': token_targets,
        'type_targets': type_targets,
        'eos_positions': eos_positions,
        'valid_counts': valid_counts,
        'normalized_targets': normalized_targets,
    }


def decode_frlp_batch(token_logits, eos_logits=None, confidence_threshold=0.5):
    token_probs = token_logits.softmax(dim=-1)
    token_confidences, token_ids = token_probs.max(dim=-1)
    batch_size, max_steps = token_ids.shape

    if eos_logits is not None:
        eos_positions = torch.argmax(eos_logits, dim=-1)
    else:
        eos_positions = torch.full((batch_size,), max_steps, dtype=torch.long, device=token_ids.device)
        for batch_idx in range(batch_size):
            eos_mask = token_ids[batch_idx] == FRLP_EOS_ID
            if torch.any(eos_mask):
                eos_positions[batch_idx] = int(torch.argmax(eos_mask.int()).item())

    texts = []
    compact_texts = []
    confidences = []
    char_confidences = []
    decoded_token_ids = []

    for batch_idx in range(batch_size):
        stop = int(eos_positions[batch_idx].item())
        stop = max(0, min(stop, max_steps))

        display_chars = []
        compact_chars = []
        per_char_conf = []
        decoded_ids = []

        for step_idx in range(stop):
            token_id = int(token_ids[batch_idx, step_idx].item())
            token_conf = float(token_confidences[batch_idx, step_idx].item())
            decoded_ids.append(token_id)

            if token_id in (FRLP_BLANK_ID, FRLP_EOS_ID) or token_conf < confidence_threshold:
                display_chars.append(' ')
                per_char_conf.append(token_conf)
                continue

            char = FRLP_ID_TO_TOKEN[token_id]
            display_chars.append(char)
            compact_chars.append(char)
            per_char_conf.append(token_conf)

        text = ''.join(display_chars).rstrip()
        texts.append(text)
        compact_texts.append(''.join(compact_chars))
        char_confidences.append(per_char_conf)
        decoded_token_ids.append(decoded_ids)
        confidences.append(sum(per_char_conf) / max(len(per_char_conf), 1))

    return {
        'texts': texts,
        'compact_texts': compact_texts,
        'confidences': torch.tensor(confidences, dtype=torch.float32),
        'char_confidences': char_confidences,
        'token_ids': token_ids.detach().cpu(),
        'decoded_token_ids': decoded_token_ids,
        'eos_positions': eos_positions.detach().cpu(),
    }
