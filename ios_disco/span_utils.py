from typing import Iterable, List, Sequence, Tuple


Span = Tuple[int, int]


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_i = len(sequence) - len(pattern)
    for i in range(start, max_i + 1):
        if list(sequence[i : i + len(pattern)]) == list(pattern):
            return i
    return -1


def unique_patterns(tokenizer, strings: Iterable[str]) -> List[List[int]]:
    patterns = []
    seen = set()
    for text in strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            patterns.append(ids)
            seen.add(key)
    return patterns


def default_start_patterns(tokenizer) -> List[List[int]]:
    patterns = unique_patterns(tokenizer, ["{{", " {{"])
    # Llama-3.x legacy tokenizer in the original notebook tokenized " {{" as 5991.
    for pattern in ([5991],):
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def default_end_patterns(tokenizer) -> List[List[int]]:
    patterns = unique_patterns(tokenizer, ["}}", " }}"])
    # Llama-3.x legacy tokenizer variants observed in the original notebook.
    for pattern in ([3500], [23742]):
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def find_first_pattern(sequence: Sequence[int], patterns: Sequence[Sequence[int]], start: int = 0):
    best = None
    for pattern in patterns:
        idx = find_subsequence(sequence, pattern, start=start)
        if idx == -1:
            continue
        if best is None or idx < best[0]:
            best = (idx, len(pattern))
    return best


def find_abc_spans(
    input_ids: Sequence[int],
    start_patterns: Sequence[Sequence[int]],
    end_patterns: Sequence[Sequence[int]],
    max_spans: int = 4,
) -> List[Span]:
    spans = []
    cursor = 0
    while cursor < len(input_ids) and len(spans) < max_spans:
        start_match = find_first_pattern(input_ids, start_patterns, start=cursor)
        if start_match is None:
            break
        start_idx, start_len = start_match

        content_start = start_idx + start_len
        end_match = find_first_pattern(input_ids, end_patterns, start=content_start)
        if end_match is None:
            break
        end_idx, end_len = end_match

        if content_start < end_idx:
            spans.append((content_start, end_idx))
        cursor = end_idx + end_len

    return spans
