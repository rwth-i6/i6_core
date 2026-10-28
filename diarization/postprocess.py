"""
Attribution of the words of transcribed whole recordings to diarized speakers, written as a corpus of
speaker turns.
"""

__all__ = ["AttributeSpeakersJob", "attribute_speakers", "speaker_turns", "words_to_turns"]

import json
import os
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

from sisyphus import Job, Task, tk

from i6_core.lib.intervals import Interval, union_intervals

Word = dict[str, Any]
Segment = tuple[str, float, float]


def words_to_turns(words: Sequence[Sequence[Word]], *, max_gap: float) -> list[Word]:
    """
    Inter-pausal units: words of a channel joined while the pause between them is at most `max_gap`.

    :param words: per channel, `{text, start, end}` sorted by start
    :return: `{channel, start, end, text}` sorted by start
    """
    turns = []
    for channel, channel_words in enumerate(words):
        turn = None
        for word in channel_words:
            if turn is not None and word["start"] - turn["end"] <= max_gap:
                turn["end"] = max(turn["end"], word["end"])
                turn["text"] += " " + word["text"]
            else:
                turn = {"channel": channel, "start": word["start"], "end": word["end"], "text": word["text"]}
                turns.append(turn)
    return sorted(turns, key=lambda t: (t["start"], t["channel"]))


def _speaker_intervals(segments: Sequence[Segment]) -> dict[str, list[Interval]]:
    """
    :param segments: diarization `(speaker, start, end)`
    :return: per speaker the union of its segments, disjoint and sorted
    """
    intervals = {}
    for speaker, start, end in segments:
        intervals.setdefault(speaker, []).append((start, end))
    return {speaker: union_intervals(v) for speaker, v in intervals.items()}


def _overlap_and_distance(intervals: Sequence[Interval], start: float, end: float) -> tuple[float, float]:
    """
    :param intervals: disjoint and sorted
    :return: the seconds of `[start, end]` covered by the intervals, and the gap to the nearest interval
        (0 if one overlaps, inf without intervals)
    """
    i = bisect_right(intervals, start, key=lambda interval: interval[1])  # first interval ending after start
    overlap = 0.0
    for interval_start, interval_end in intervals[i:]:
        if interval_start >= end:
            break
        overlap += min(end, interval_end) - max(start, interval_start)
    before = start - intervals[i - 1][1] if i > 0 else float("inf")
    after = max(0.0, intervals[i][0] - end) if i < len(intervals) else float("inf")
    return overlap, min(before, after)


def attribute_speakers(
    words: Sequence[Sequence[Word]],
    segments: Sequence[Segment],
    *,
    channel_speakers: Sequence[Sequence[str]] | None = None,
) -> list[Word]:
    """
    Each word gets the candidate speaker whose diarized speech overlaps it most, or the nearest one if
    none overlaps; a channel without candidates is its own speaker. Speakers are renamed `S1`, `S2`,
    ... in order of first appearance.

    :param words: per channel, `{text, start, end, ...}`
    :param segments: diarization `(speaker, start, end)`
    :param channel_speakers: per channel the candidate speakers; None for all diarized speakers
    :return: the words of all channels sorted by start, with `speaker`, `channel` and `overlap`
        (whether diarized speech of the speaker overlaps the word)
    """
    intervals = _speaker_intervals(segments)

    result = []
    for channel, channel_words in enumerate(words):
        if channel_speakers is None:
            candidates = sorted(intervals)
        else:
            candidates = sorted(s for s in channel_speakers[channel] if s in intervals)
        for word in channel_words:
            if candidates:
                scores = {s: _overlap_and_distance(intervals[s], word["start"], word["end"]) for s in candidates}
                speaker = max(candidates, key=lambda s: (scores[s][0], -scores[s][1]))  # most overlap, then nearest
                overlap = scores[speaker][0] > 0
            else:
                speaker, overlap = f"ch{channel}", False
            result.append({**word, "speaker": speaker, "channel": channel, "overlap": overlap})
    result.sort(key=lambda w: (w["start"], w["channel"]))

    names = {}
    for word in result:
        word["speaker"] = names.setdefault(word["speaker"], f"S{len(names) + 1}")
    return result


def _separated(a: Word, b: Word, barriers: Sequence[Interval]) -> bool:
    """
    :param barriers: disjoint and sorted
    :return: whether a barrier reaches into the time between the words
    """
    i = bisect_right(barriers, a["end"], key=lambda barrier: barrier[1])  # first barrier ending after a
    return i < len(barriers) and barriers[i][0] < b["start"]


def _split_at_barriers(words: Sequence[Word], barriers: Sequence[Interval]) -> list[list[Word]]:
    """
    :param words: sorted by start
    :return: the words split between consecutive ones separated by a barrier
    """
    groups = []
    for word in words:
        if not groups or _separated(groups[-1][-1], word, barriers):
            groups.append([])
        groups[-1].append(word)
    return groups


def _turn(words: Sequence[Word]) -> Word:
    """
    :param words: of one speaker, sorted by start
    :return: `{start, end, speaker, channel, words}`, `channel` the majority channel
    """
    from collections import Counter

    return {
        "start": words[0]["start"],
        "end": max(w["end"] for w in words),
        "speaker": words[0]["speaker"],
        "channel": Counter(w["channel"] for w in words).most_common(1)[0][0],
        "words": list(words),
    }


def speaker_turns(words: Sequence[Word], *, gap_seconds: float) -> list[Word]:
    """
    Turns of each speaker: its words, split where the other speakers speak for more than `gap_seconds`
    (their words joined at pauses up to `gap_seconds`); shorter interjections stay inside the turn.

    :param words: `{text, start, end, speaker, channel}` sorted by start
    :return: `{start, end, speaker, channel, words}` sorted by start, `channel` the majority channel
    """
    turns = []
    for speaker in sorted({w["speaker"] for w in words}):
        own = [w for w in words if w["speaker"] == speaker]
        others = [w for w in words if w["speaker"] != speaker]
        other_speech = words_to_turns([others], max_gap=gap_seconds)
        barriers = [(u["start"], u["end"]) for u in other_speech if u["end"] - u["start"] > gap_seconds]
        turns.extend(_turn(group) for group in _split_at_barriers(own, barriers))
    return sorted(turns, key=lambda t: (t["start"], t["speaker"]))


def _channel_speakers(words: Sequence[Sequence[Word]], segments: Sequence[Segment]) -> list[list[str]]:
    """
    :return: per channel the diarized speakers whose speech overlaps the words of that channel most
    """
    intervals = _speaker_intervals(segments)
    overlaps = {speaker: [0.0] * len(words) for speaker in intervals}  # seconds per speaker and channel
    for channel, channel_words in enumerate(words):
        for word in channel_words:
            for speaker, speaker_intervals in intervals.items():
                overlaps[speaker][channel] += _overlap_and_distance(speaker_intervals, word["start"], word["end"])[0]
    result = [[] for _ in words]
    for speaker, per_channel in overlaps.items():
        if any(per_channel):
            channel = max(range(len(words)), key=lambda c: (per_channel[c], -c))  # ties to the lower channel
            result[channel].append(speaker)
    return result


def _recording_words(tracks: dict[int, list[tuple[str, float, float]]], *, per_channel: bool) -> list[list[Word]]:
    """
    :param tracks: CTM `(word, start, end)` per track, the track being the channel + 1
    :return: per channel `{text, start, end}` sorted by start; all words in one channel if not `per_channel`
    """
    if per_channel:
        channels = [tracks.get(channel + 1, []) for channel in range(max(tracks, default=0))]
    else:
        channels = [sorted((w for track_words in tracks.values() for w in track_words), key=lambda w: w[1:])]
    return [[{"text": text, "start": start, "end": end} for text, start, end in channel] for channel in channels]


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    """
    :return: count, mean and percentiles of the values, or None without values
    """
    import numpy as np

    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        **{f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)},
    }


class AttributeSpeakersJob(Job):
    """
    Attributes the words of recognized whole recordings to the speakers of the diarization (see
    `attribute_speakers`) and writes a corpus whose segments are the speaker turns (see
    `speaker_turns`). On recordings recognized per channel (segments with `track`), the candidates
    of a channel are the speakers whose speech overlaps the words of that channel most.

    `corpus.xml.gz` holds per recording one segment per turn with the speaker `<recording>_S1`,
    `S2`, ..., the `track` of the turn words on per-channel recordings, and the words of the
    speaker as orth. `summary.json` holds the words without overlapping diarized speech, the
    speakers per recording before and after the attribution and the turns per recording.
    """

    def __init__(self, *, corpus: tk.Path, ctms: list[tk.Path], rttm_dir: tk.Path, turn_gap_seconds: float = 0.5):
        """
        :param corpus: bliss corpus of the recognized recordings
        :param ctms: word-level CTMs of the recordings, the track column being the channel + 1
        :param rttm_dir: `rttm/<name>.rttm` per recording
        :param turn_gap_seconds: maximal pause within speech of another speaker and its minimal
            duration to end a turn
        """
        self.corpus = corpus
        self.ctms = ctms
        self.rttm_dir = rttm_dir
        self.turn_gap_seconds = turn_gap_seconds

        self.out_corpus = self.output_path("corpus.xml.gz")
        self.out_summary = self.output_path("summary.json")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from collections import Counter

        import i6_core.lib.corpus as libcorpus
        from i6_core.lib.ctm import parse_ctm_words
        from i6_core.lib.rttm import parse_rttm

        ctm_words = {}
        for ctm in self.ctms:
            for name, tracks in parse_ctm_words(ctm).items():
                ctm_words.setdefault(name, {}).update(tracks)

        bliss_corpus = libcorpus.Corpus()
        bliss_corpus.load(self.corpus.get_path())
        totals = Counter()
        speaker_counts = Counter()
        turns_per_recording = []
        for recording in bliss_corpus.all_recordings():
            tracked = any(segment.track is not None for segment in recording.segments)
            words = _recording_words(ctm_words.get(recording.name, {}), per_channel=tracked)
            diarization = parse_rttm(os.path.join(self.rttm_dir.get_path(), f"{recording.name}.rttm"))
            channel_speakers = _channel_speakers(words, diarization) if tracked else None
            attributed = attribute_speakers(words, diarization, channel_speakers=channel_speakers)
            turns = speaker_turns(attributed, gap_seconds=self.turn_gap_seconds)

            recording.segments = []
            for i, turn in enumerate(turns, start=1):
                speaker_name = f"{recording.name}_{turn['speaker']}"
                if speaker_name not in bliss_corpus.speakers:
                    speaker = libcorpus.Speaker()
                    speaker.name = speaker_name
                    bliss_corpus.add_speaker(speaker)
                segment = libcorpus.Segment(
                    start=turn["start"],
                    end=turn["end"],
                    track=turn["channel"] if tracked else None,
                    orth=" ".join(w["text"] for w in turn["words"]),
                    speaker_name=speaker_name,
                )
                segment.name = str(i)
                recording.add_segment(segment)

            num_diarized = len({speaker for speaker, _, _ in diarization})
            num_attributed = len({w["speaker"] for w in attributed})
            speaker_counts[f"{num_diarized}->{num_attributed}"] += 1
            turns_per_recording.append(len(turns))
            totals["recordings"] += 1
            totals["words"] += len(attributed)
            totals["words_without_overlap"] += sum(not w["overlap"] for w in attributed)
            totals["turns"] += len(turns)
            totals["turn_seconds"] += sum(t["end"] - t["start"] for t in turns)
        bliss_corpus.dump(self.out_corpus.get_path())

        summary = {
            **totals,
            "turn_hours": totals["turn_seconds"] / 3600,
            "words_without_overlap_share": totals["words_without_overlap"] / max(totals["words"], 1),
            "speakers_diarized_to_attributed": dict(speaker_counts.most_common()),
            "turns_per_recording": _distribution(turns_per_recording),
        }
        with open(self.out_summary.get_path(), "wt") as f:
            json.dump(summary, f, indent=2)
