__all__ = ["parse_ctm_words"]

from sisyphus import tk

from i6_core.util import uopen


def parse_ctm_words(ctm_file: tk.Path | str) -> dict[str, dict[int, list[tuple[str, float, float]]]]:
    """
    :param ctm_file: CTM with `<name> <track> <start> <duration> <word> ...` lines, comments ignored
    :return: per recording and track the `(word, start, end)` of its lines, sorted by start
    """
    words: dict[str, dict[int, list[tuple[str, float, float]]]] = {}
    with uopen(ctm_file, "rt") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0].startswith(";;"):
                continue
            name, track, start, duration, word = fields[:5]
            start = float(start)
            words.setdefault(name, {}).setdefault(int(track), []).append((word, start, start + float(duration)))
    for tracks in words.values():
        for track_words in tracks.values():
            track_words.sort(key=lambda w: (w[1], w[2]))
    return words
