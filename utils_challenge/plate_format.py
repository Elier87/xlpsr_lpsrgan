LETTER_POS_MAP = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'Z',
    '8': 'B',
}

DIGIT_POS_MAP = {
    'A': '4',
    'B': '8',
    'D': '0',
    'G': '6',
    'I': '1',
    'J': '1',
    'O': '0',
    'Q': '0',
    'S': '5',
    'T': '1',
    'Z': '7',
}

FRANCE_PATTERNS = (
    'LLDDDLL',
    'DDDDLLDD',
)


def normalize_plate_text(text):
    return ''.join(ch for ch in (text or '').upper() if ch.isalnum())


def _repair_to_pattern(text, pattern):
    if len(text) != len(pattern):
        return None, None

    repaired = []
    edits = 0
    for char, slot in zip(text, pattern):
        if slot == 'L':
            fixed = LETTER_POS_MAP.get(char, char)
            if not fixed.isalpha():
                return None, None
        else:
            fixed = DIGIT_POS_MAP.get(char, char)
            if not fixed.isdigit():
                return None, None
        edits += int(fixed != char)
        repaired.append(fixed)
    return ''.join(repaired), edits


def apply_french_plate_format(text):
    text = normalize_plate_text(text)
    if not text:
        return text

    candidates = []
    for pattern in FRANCE_PATTERNS:
        repaired, edits = _repair_to_pattern(text, pattern)
        if repaired is not None:
            candidates.append((edits, repaired))

    if not candidates:
        return text

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def apply_plate_format(text, enabled=False, country='fr'):
    if not enabled:
        return normalize_plate_text(text)

    if country.lower() == 'fr':
        return apply_french_plate_format(text)

    return normalize_plate_text(text)
