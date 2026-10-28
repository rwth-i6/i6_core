__all__ = ["parse_rttm"]

from sisyphus import tk

from i6_core.util import uopen


def parse_rttm(rttm_file: tk.Path | str) -> list[tuple[str, float, float]]:
    """
    :param rttm_file: RTTM with `SPEAKER <name> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>` lines
    :return: `(speaker, start, end)` of the SPEAKER lines, sorted by start
    """
    segments = []
    with uopen(rttm_file, "rt") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            start, duration = float(fields[3]), float(fields[4])
            segments.append((fields[7], start, start + duration))
    return sorted(segments, key=lambda s: (s[1], s[2]))
