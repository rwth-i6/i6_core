"""
Rich Transcription Time Marked (RTTM) files, as written by speaker diarization systems.
"""

from __future__ import annotations

__all__ = ["RttmSpeakerTurn", "parse_rttm"]

from dataclasses import dataclass
from typing import List, Optional, Union

from sisyphus import tk

from i6_core.util import uopen


@dataclass(frozen=True)
class RttmSpeakerTurn:
    """
    One `SPEAKER` line of an RTTM file:
    `SPEAKER <file> <channel> <start> <duration> <orthography> <speaker_type> <speaker> <confidence> <lookahead>`,
    with `<NA>` fields as `None`.
    """

    file: str
    channel: int
    start: float
    duration: float
    speaker: str
    orthography: Optional[str] = None
    speaker_type: Optional[str] = None
    confidence: Optional[float] = None
    lookahead: Optional[float] = None

    @property
    def end(self) -> float:
        return self.start + self.duration


def _optional(field: Optional[str]) -> Optional[str]:
    return None if field is None or field == "<NA>" else field


def _optional_float(field: Optional[str]) -> Optional[float]:
    field = _optional(field)
    return None if field is None else float(field)


def parse_rttm(rttm_file: Union[tk.Path, str]) -> List[RttmSpeakerTurn]:
    """
    :param rttm_file: RTTM file, possibly gzipped; lines other than `SPEAKER` lines are ignored
    :return: the `SPEAKER` lines of all files and channels in the RTTM, sorted by start and end
    """
    turns = []
    with uopen(rttm_file, "rt") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            assert len(fields) >= 8, f"malformed RTTM line: {line!r}"
            fields += [None] * (10 - len(fields))
            turns.append(
                RttmSpeakerTurn(
                    file=fields[1],
                    channel=int(fields[2]),
                    start=float(fields[3]),
                    duration=float(fields[4]),
                    orthography=_optional(fields[5]),
                    speaker_type=_optional(fields[6]),
                    speaker=fields[7],
                    confidence=_optional_float(fields[8]),
                    lookahead=_optional_float(fields[9]),
                )
            )
    return sorted(turns, key=lambda t: (t.start, t.end))
