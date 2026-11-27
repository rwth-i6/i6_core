__all__ = ["Interval", "split_pieces", "union_intervals"]

Interval = tuple[float, float]


def union_intervals(intervals: list[Interval]) -> list[Interval]:
    """
    :return: the merged intervals, disjoint and sorted
    """
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split_pieces(intervals: list[Interval], *, duration: float, max_seconds: float) -> list[Interval]:
    """
    Splits `[0, duration]` into consecutive pieces of at most `max_seconds`. Each piece ends in the
    middle of the longest gap between the intervals in its second half, or at `max_seconds` if
    there is no gap there, so all pieces but the last are at least `max_seconds / 2` long.

    :param intervals: `(start, end)` not to be cut if possible, e.g. the speech of a diarization
    :return: `(start, end)` per piece, covering `[0, duration]`
    """
    occupied = union_intervals(intervals)
    gaps = [(a_end, b_start) for (_, a_end), (b_start, _) in zip(occupied, occupied[1:])]
    pieces, start = [], 0.0
    while duration - start > max_seconds:
        lower, upper = start + max_seconds / 2, start + max_seconds
        candidates = [(b - a, (a + b) / 2) for a, b in gaps if lower <= (a + b) / 2 <= upper]
        cut = max(candidates)[1] if candidates else upper
        pieces.append((start, cut))
        start = cut
    pieces.append((start, duration))
    return pieces
