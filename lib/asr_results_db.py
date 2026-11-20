"""
Compact SQLite storage for ASR results ("ASR results DB"), readable as the models of :mod:`i6_core.lib.asr_results`.

The JSONL format of :mod:`i6_core.lib.asr_results` stores one JSON object per word, which is too large and too slow
to parse for corpora with billions of words. This format keeps the recordings and segments as rows that can be
queried with SQL, and stores the words of each segment column-wise in (optionally compressed) blobs that are cheap to
write from C/C++ and cheap to decode with numpy.

Format, version 1
=================

A file is an SQLite 3 database with ``PRAGMA application_id = 0x41535244`` ("ASRD") and
``PRAGMA user_version = 1`` (the format version). Readers must reject other application ids and unknown versions.

Tables::

    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE speakers (
        id INTEGER PRIMARY KEY,            -- >= 1
        speaker_id TEXT NOT NULL UNIQUE,   -- SpeakerInfo.id
        name TEXT,                         -- SpeakerInfo.name
        gender TEXT                        -- SpeakerInfo.gender
    );
    CREATE TABLE languages (
        id INTEGER PRIMARY KEY,            -- >= 1
        lang_code TEXT NOT NULL UNIQUE     -- TimedTranscript.lang_code
    );
    CREATE TABLE recordings (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,         -- Recording.name
        path TEXT,                         -- Recording.path; with meta "audio_root": relative to it, or absolute
        status TEXT NOT NULL DEFAULT 'OK', -- e.g. 'OK', 'ASR_ERROR'; not part of the i6_core models
        error TEXT,                        -- error message if status is not 'OK'
        duration_ms INTEGER                -- duration of the audio, if known
    );
    CREATE TABLE segments (
        id INTEGER PRIMARY KEY,            -- ascending in the order of the segments in the recording
        recording_id INTEGER NOT NULL REFERENCES recordings(id),
        name TEXT NOT NULL,                -- Segment.name; n-best alternatives share the name
        n_best_index INTEGER,              -- Segment.n_best_index
        start_ms INTEGER,                  -- Segment.start
        end_ms INTEGER,                    -- Segment.end
        speaker INTEGER REFERENCES speakers(id),  -- Segment.speaker
        num_tokens INTEGER NOT NULL,       -- number of TimedTranscripts of the segment
        tokens BLOB,                       -- TimedTranscript.text
        start_deltas BLOB,                 -- TimedTranscript.start
        durations BLOB,                    -- TimedTranscript.end - TimedTranscript.start
        channels BLOB,                     -- TimedTranscript.channel
        lang_codes BLOB,                   -- TimedTranscript.lang_code
        token_speakers BLOB,               -- TimedTranscript.speaker
        confidences BLOB,                  -- TimedTranscript.confidence
        am_scores BLOB,                    -- TimedTranscript.am_score
        lm_scores BLOB                     -- TimedTranscript.lm_score
    );
    CREATE INDEX segments_recording ON segments (recording_id, id);

Rules:

* Every field of the i6_core models is stored where the model has it: Segment fields as columns of ``segments``,
  TimedTranscript fields per token.
* All times are integer milliseconds from the beginning of the recording: ``seconds * 1000`` computed in double
  precision and rounded half to even (C ``rint`` in the default rounding mode, ``numpy.rint``, Python ``round``).
* The segments of a recording are its rows of ``segments`` in ``id`` order. Rows of different recordings may be
  interleaved, e.g. when results of several recordings are written as they come in; readers group them by
  ``recording_id``. Every segment must belong to a recording.
* Segments of a recording with the same name next to each other are the n-best alternatives of one segment (as in
  the JSONL format), ordered by ``n_best_index`` (None first) when read as i6_core models. Empty groups, groups
  mixing names and neighbouring groups of the same name cannot be represented; writers reject them.
* Per-token blobs, each holding ``num_tokens`` values in token order. All are NULL if ``num_tokens`` is 0. Small
  deltas and durations compress about twice as well as absolute times.

  - ``tokens``: the UTF-8 texts joined by NUL bytes (0x00), with no trailing separator.
  - ``start_deltas``: little-endian uint32, the start of each token minus the start of the previous token (the first
    one minus 0, i.e. its start), modulo 2**32: the start of a token is the start of the previous one plus its delta,
    wrapping around at 2**32, as uint32 arithmetic does in C/C++ and numpy. A token that starts before the previous
    one (tokens need not be ordered by start) has a delta of ``2**32 - difference``.
  - ``durations``: little-endian uint32, end minus start of each token. The end ``start + duration`` is computed
    with at least 33 bits.

  With unsigned values, every stored start and duration is valid, so readers need no range checks for them.
  - ``channels``: little-endian uint16, the 1-based channel. NULL if all tokens are on channel 1.
  - ``lang_codes``: little-endian uint16, ``languages.id`` of the token's lang_code, 0 if it has none. NULL if no
    token has one.
  - ``token_speakers``: little-endian uint32, ``speakers.id`` of the token's speaker, 0 if it has none. NULL if no
    token has one.
  - ``confidences``: little-endian uint16 holding ``confidence * 10000`` rounded half to even (0..10000), 65535 if the token
    has no confidence. NULL if no token has one.
  - ``am_scores``, ``lm_scores``: little-endian float32, NaN if the token has no score. NULL if no token has one.

  A blob that is NULL for "all default" may also be written with the default values; readers must accept both.

* Compression: if meta ``codec`` is ``zstd``, every non-NULL blob is one zstd frame. Writers may use any
  compression level and should write the content size into the frame header. If ``codec`` is ``none``, the blobs
  are stored as is.
* Required meta keys: ``codec``. Optional meta keys: ``audio_root`` (an absolute directory; if set, recording paths
  below it are stored relative to it and all others absolute, so readers join relative paths onto it) and
  free-form ones describing how the results were produced, e.g. ``producer``, ``model``, ``language``,
  ``description``.
* Limits, which writers enforce by raising an error:

  - times, including token ends, 0 .. 4294967295 ms (uint32, about 49.7 days); not infinite;
  - channels 1 .. 65535; ``n_best_index`` 0 .. 2**63 - 1; at most 65535 distinct lang_codes;
  - finite am/lm scores within float32 range (about +-3.4e38); infinite scores are stored as such;
  - texts must be valid UTF-8 (no lone surrogates) and must not contain NUL.

* Readers must reject as format errors: per-token blobs whose number of values differs from ``num_tokens``, segment
  ``start_ms`` / ``end_ms`` outside the limits, confidence codes outside 0..10000 and 65535, channels 0, and
  lang_code or speaker indices that do not exist. Token ends above the limit (up to ``2 * (2**32 - 1)``, which a
  writer never produces) need not be rejected.
* Files are written once and then only read. Readers may open them with ``?immutable=1``.
"""

__all__ = [
    "FORMAT_VERSION",
    "APPLICATION_ID",
    "SegmentArrays",
    "RecordingInfo",
    "ASRResultsDBWriter",
    "ASRResultsDBReader",
]

import math
import os
import sqlite3
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Optional

import numpy as np

from i6_core.lib.asr_results import BatchRecording, BatchSegment, SpeakerInfo

FORMAT_VERSION = 1
APPLICATION_ID = 0x41535244  # "ASRD"

_CONFIDENCE_SCALE = 10000
_CONFIDENCE_NONE = 0xFFFF
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE speakers (
    id INTEGER PRIMARY KEY,
    speaker_id TEXT NOT NULL UNIQUE,
    name TEXT,
    gender TEXT
);
CREATE TABLE languages (
    id INTEGER PRIMARY KEY,
    lang_code TEXT NOT NULL UNIQUE
);
CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT,
    status TEXT NOT NULL DEFAULT 'OK',
    error TEXT,
    duration_ms INTEGER
);
CREATE TABLE segments (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    name TEXT NOT NULL,
    n_best_index INTEGER,
    start_ms INTEGER,
    end_ms INTEGER,
    speaker INTEGER REFERENCES speakers(id),
    num_tokens INTEGER NOT NULL,
    tokens BLOB,
    start_deltas BLOB,
    durations BLOB,
    channels BLOB,
    lang_codes BLOB,
    token_speakers BLOB,
    confidences BLOB,
    am_scores BLOB,
    lm_scores BLOB
);
CREATE INDEX segments_recording ON segments (recording_id, id);
"""


def _zstd_compressor(level: int):
    try:
        from compression import zstd  # Python >= 3.14

        return lambda data: zstd.compress(data, level=level)
    except ImportError:
        import zstandard

        return zstandard.ZstdCompressor(level=level, write_content_size=True).compress


def _zstd_decompressor():
    try:
        from compression import zstd  # Python >= 3.14

        return zstd.decompress
    except ImportError:
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        # decompressobj also handles frames without content size
        return lambda data: dctx.decompressobj().decompress(data)


def _to_ms(seconds: Optional[float]) -> Optional[int]:
    if seconds is None:
        return None
    product = seconds * 1000
    if not math.isfinite(product):
        raise ValueError(f"time {seconds} s is not finite or does not fit into uint32 milliseconds")
    ms = round(product)  # half to even
    if not 0 <= ms <= _UINT32_MAX:
        raise ValueError(f"time {seconds} s does not fit into uint32 milliseconds")
    return ms


def _from_ms(ms: Optional[int]) -> Optional[float]:
    return None if ms is None else ms / 1000


@dataclass
class SegmentArrays:
    """
    One segment with its tokens as arrays: the unit in which segments are stored, and the fast way to read them.

    Times are in seconds for the segment and in integer milliseconds for the tokens. The per-token fields after
    ``ends_ms`` are None if all tokens have the default (channel 1, no language/speaker/confidence/score);
    otherwise they hold one value per token, missing confidences and scores being NaN.
    """

    name: str
    tokens: list[str]
    starts_ms: np.ndarray  # uint32 when read
    ends_ms: np.ndarray  # int64 when read
    channels: Optional[np.ndarray] = None  # 1-based, None = all 1
    lang_codes: Optional[list[Optional[str]]] = None
    token_speakers: Optional[list[Optional[SpeakerInfo]]] = None
    confidences: Optional[np.ndarray] = None  # float, NaN = none
    am_scores: Optional[np.ndarray] = None  # float32, NaN = none
    lm_scores: Optional[np.ndarray] = None  # float32, NaN = none
    n_best_index: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[SpeakerInfo] = None

    @classmethod
    def from_batch_segment(cls, segment: BatchSegment) -> "SegmentArrays":
        transcripts = segment.transcripts
        channels = [tt.channel for tt in transcripts]
        lang_codes = [tt.lang_code for tt in transcripts]
        token_speakers = [tt.speaker for tt in transcripts]
        return cls(
            name=segment.name,
            n_best_index=segment.n_best_index,
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker,
            channels=None if all(c == 1 for c in channels) else np.array(channels, dtype=np.int64),
            lang_codes=None if all(c is None for c in lang_codes) else lang_codes,
            token_speakers=None if all(s is None for s in token_speakers) else token_speakers,
            tokens=[tt.text for tt in transcripts],
            starts_ms=np.array([_to_ms(tt.start) for tt in transcripts], dtype=np.int64),
            ends_ms=np.array([_to_ms(tt.end) for tt in transcripts], dtype=np.int64),
            confidences=_optional_array([tt.confidence for tt in transcripts], np.float64),
            am_scores=_optional_array([tt.am_score for tt in transcripts], np.float32),
            lm_scores=_optional_array([tt.lm_score for tt in transcripts], np.float32),
        )

    def to_batch_segment(self) -> BatchSegment:
        """
        :return: the segment as i6_core model; fields that are None are left unset
        """
        transcripts = []
        starts = self.starts_ms.tolist()
        ends = self.ends_ms.tolist()
        channels = None if self.channels is None else self.channels.tolist()
        confidences = None if self.confidences is None else self.confidences.tolist()
        am_scores = None if self.am_scores is None else self.am_scores.tolist()
        lm_scores = None if self.lm_scores is None else self.lm_scores.tolist()
        for i, text in enumerate(self.tokens):
            values = {"text": text, "start": starts[i] / 1000, "end": ends[i] / 1000}
            # the default channel is left unset, so that JSONL written with exclude_unset omits it as for the original
            if channels is not None and channels[i] != 1:
                values["channel"] = channels[i]
            if self.lang_codes is not None and self.lang_codes[i] is not None:
                values["lang_code"] = self.lang_codes[i]
            if self.token_speakers is not None and self.token_speakers[i] is not None:
                values["speaker"] = self.token_speakers[i]
            for key, array in (("confidence", confidences), ("am_score", am_scores), ("lm_score", lm_scores)):
                if array is not None and not math.isnan(array[i]):
                    values[key] = array[i]
            transcripts.append(values)
        values = {"name": self.name, "transcripts": transcripts}
        for key in ("n_best_index", "start", "end", "speaker"):
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        # validating the whole segment at once is ~3x faster than model_construct per transcript
        return BatchSegment.model_validate(values)


def _optional_array(values: Sequence[Optional[float]], dtype) -> Optional[np.ndarray]:
    if all(v is None for v in values):
        return None
    return np.array([math.nan if v is None else v for v in values], dtype=dtype)


@dataclass
class RecordingInfo:
    """
    A recording and the fields of this format that the i6_core models do not have.
    """

    name: str
    path: Optional[str] = None
    status: str = "OK"
    error: Optional[str] = None
    duration: Optional[float] = None  # seconds
    segments: list[SegmentArrays] = field(default_factory=list)

    def to_batch_recording(self) -> BatchRecording:
        """
        :return: the recording as i6_core model, the segments with the same name grouped as n-best alternatives
            sorted by n-best index (None first), as in :func:`i6_core.lib.asr_results.parse_asr_results`
        """
        alternatives: list[list[BatchSegment]] = []
        for segment in self.segments:
            batch_segment = segment.to_batch_segment()
            if alternatives and alternatives[-1][-1].name == batch_segment.name:
                alternatives[-1].append(batch_segment)
            else:
                alternatives.append([batch_segment])
        for alternative in alternatives:
            alternative.sort(key=lambda s: s.n_best_index if s.n_best_index is not None else -1)
        values = {"name": self.name, "alternatives": []}
        if self.path is not None:
            values["path"] = self.path
        # validate the recording fields (path becomes the same type as when parsing JSONL), not the segments again
        recording = BatchRecording.model_validate(values)
        recording.alternatives = alternatives
        return recording


class ASRResultsDBWriter:
    """
    Writes a new ASR results DB. Use as context manager, the file is complete once the context is left without error::

        with ASRResultsDBWriter("results.db", meta={"model": "..."}) as writer:
            for recording in recordings:
                writer.add_recording(recording)

    For files on network file systems, write to local storage and move the finished file.
    """

    def __init__(
        self,
        path: str | PathLike,
        *,
        codec: str = "zstd",
        compression_level: int = 3,
        meta: Optional[Mapping[str, str]] = None,
        audio_root: Optional[str] = None,
    ):
        """
        :param path: the file to create, must not exist
        :param codec: "zstd" or "none"
        :param compression_level: zstd compression level, 1..19 (higher is smaller and slower)
        :param meta: additional meta entries, e.g. producer, model, language
        :param audio_root: absolute directory; if set, recording paths are made absolute (relative ones against the
            current directory), and those below it are stored relative to it
        """
        if codec not in ("zstd", "none"):
            raise ValueError(f"unknown codec {codec!r}")
        if audio_root is not None:
            if not os.path.isabs(audio_root):
                raise ValueError(f"audio_root must be absolute: {audio_root}")
            audio_root = os.path.normpath(audio_root)
        if os.path.exists(path):
            raise FileExistsError(path)
        self.path = os.fspath(path)
        self._compress = _zstd_compressor(compression_level) if codec == "zstd" else None
        self._audio_root = audio_root
        # {name: (id, path, status, error, duration_ms, name of its last segment)} of the recordings written so far
        self._recordings: dict[str, tuple] = {}
        self._speaker_ids: dict[str, tuple[int, SpeakerInfo]] = {}
        self._language_ids: dict[str, int] = {}

        # transactions are controlled explicitly: everything is written in one transaction
        self._db = sqlite3.connect(self.path, isolation_level=None)
        # the rollback journal is needed for the per-recording savepoints, but never needs to survive a crash:
        # a failed write deletes the file
        self._db.execute("PRAGMA journal_mode = MEMORY")
        self._db.execute("PRAGMA synchronous = OFF")
        self._db.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        self._db.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
        self._db.executescript(_SCHEMA)
        entries = {"codec": codec, **(meta or {})}
        if audio_root is not None:
            entries["audio_root"] = audio_root
        self._db.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", entries.items())
        self._db.execute("BEGIN")

    def __enter__(self) -> "ASRResultsDBWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._db.close()
            os.remove(self.path)

    def close(self):
        """
        Commits all recordings and closes the file.
        """
        self._db.execute("COMMIT")
        self._db.execute("ANALYZE")
        self._db.close()

    def add_recording(self, recording: BatchRecording, **kwargs):
        """
        :param recording: the recording with all its segments
        :param kwargs: status, error, duration of :meth:`add_recording_arrays`
        :raises ValueError: for n-best structures that a flat sequence of segments cannot represent: an empty
            group, a group mixing segment names, or two neighbouring groups with the same segment name
        """
        segments = []
        previous_name = None
        for alternative in recording.alternatives:
            if not alternative:
                raise ValueError(f"recording {recording.name}: empty group of n-best alternatives")
            group_name = alternative[0].name
            if any(segment.name != group_name for segment in alternative):
                raise ValueError(f"recording {recording.name}: n-best group mixes segment names")
            if group_name == previous_name:
                raise ValueError(f"recording {recording.name}: neighbouring n-best groups of segment {group_name}")
            previous_name = group_name
            segments.extend(SegmentArrays.from_batch_segment(segment) for segment in alternative)
        path = None if recording.path is None else os.fspath(recording.path)
        self.add_recording_arrays(recording.name, segments, path=path, **kwargs)

    def add_recording_arrays(
        self,
        name: str,
        segments: Iterable[SegmentArrays],
        *,
        path: Optional[str] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None,
    ):
        """
        Adds a recording given as arrays, without going through the i6_core models.

        A recording may be added in several calls, e.g. to write the results of several recordings as they come in:
        a call with the name of a recording that was already added appends the segments to it. The other arguments
        of such a call must be None or equal to the stored values.

        :param name: name of the recording
        :param segments: the segments in recording order, n-best alternatives of a segment next to each other
        :param path: audio file of the recording
        :param status: what went wrong; None for "OK"
        :param error: error message for a status other than "OK"
        :param duration: duration of the audio in seconds
        """
        segments = list(segments)
        stored_path = self._stored_path(path)
        duration_ms = _to_ms(duration)
        # all or nothing: a rejected call leaves no rows behind, also no speakers or languages it introduced
        speaker_ids, language_ids = dict(self._speaker_ids), dict(self._language_ids)
        self._db.execute("SAVEPOINT recording")
        try:
            known = self._recordings.get(name)
            if known is None:
                cursor = self._db.execute(
                    "INSERT INTO recordings (name, path, status, error, duration_ms) VALUES (?, ?, ?, ?, ?)",
                    (name, stored_path, status or "OK", error, duration_ms),
                )
                recording_id, last_segment = cursor.lastrowid, None
            else:
                recording_id, *stored, last_segment = known
                for given, value, what in zip((stored_path, status, error, duration_ms), stored, _RECORDING_FIELDS):
                    if given is not None and given != value:
                        raise ValueError(f"recording {name} added again with a different {what}: {given!r}, {value!r}")
                if segments and segments[0].name == last_segment:
                    raise ValueError(
                        f"recording {name} continued with segment {last_segment}, which would join the n-best "
                        "alternatives written before"
                    )
            self._db.executemany(
                "INSERT INTO segments (recording_id, name, n_best_index, start_ms, end_ms, speaker, num_tokens, "
                "tokens, start_deltas, durations, channels, lang_codes, token_speakers, confidences, am_scores, "
                "lm_scores) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._segment_row(recording_id, segment) for segment in segments),
            )
        except BaseException:
            self._db.execute("ROLLBACK TO recording")
            self._db.execute("RELEASE recording")
            self._speaker_ids, self._language_ids = speaker_ids, language_ids
            raise
        self._db.execute("RELEASE recording")
        if known is None:
            self._recordings[name] = (recording_id, stored_path, status or "OK", error, duration_ms, last_segment)
        if segments:
            self._recordings[name] = (*self._recordings[name][:-1], segments[-1].name)

    def _stored_path(self, path: Optional[str]) -> Optional[str]:
        """
        :return: the path as stored: unchanged without audio_root, else absolute or relative to audio_root
        """
        if path is None or self._audio_root is None:
            return path
        path = os.path.abspath(path)
        prefix = self._audio_root.rstrip(os.sep) + os.sep
        return path[len(prefix) :] if path.startswith(prefix) else path

    def _segment_row(self, recording_id: int, segment: SegmentArrays) -> tuple:
        num_tokens = len(segment.tokens)
        for name in _PER_TOKEN_FIELDS:
            values = getattr(segment, name)
            if values is not None and len(values) != num_tokens:
                raise ValueError(f"segment {segment.name}: {name} has {len(values)} values for {num_tokens} tokens")
        if segment.n_best_index is not None and not 0 <= segment.n_best_index < 2**63:
            raise ValueError(f"segment {segment.name}: n_best_index {segment.n_best_index} out of range")
        if num_tokens == 0:
            blobs = (None,) * 9
        else:
            if any("\0" in token for token in segment.tokens):
                raise ValueError(f"segment {segment.name}: tokens must not contain NUL characters")
            try:
                tokens = "\0".join(segment.tokens).encode("utf-8")
            except UnicodeEncodeError as e:
                raise ValueError(f"segment {segment.name}: tokens are not valid unicode: {e}") from None
            starts = np.asarray(segment.starts_ms, dtype=np.int64)
            ends = np.asarray(segment.ends_ms, dtype=np.int64)
            if starts.min() < 0 or ends.max() > _UINT32_MAX or np.any(starts > ends):
                raise ValueError(f"segment {segment.name}: token times out of range or start > end")
            channels = None
            if segment.channels is not None:
                channels = np.asarray(segment.channels, dtype=np.int64)
                if channels.min() < 1 or channels.max() > 0xFFFF:
                    raise ValueError(f"segment {segment.name}: channels must be 1-based and < 65536")
                channels = channels.astype("<u2").tobytes()
            lang_codes = None
            if segment.lang_codes is not None:
                lang_codes = np.array([self._language_row(c) for c in segment.lang_codes], dtype="<u2").tobytes()
            token_speakers = None
            if segment.token_speakers is not None:
                token_speakers = np.array(
                    [self._speaker_row(s) or 0 for s in segment.token_speakers], dtype="<u4"
                ).tobytes()
            blobs = (
                tokens,
                (np.diff(starts, prepend=0) & _UINT32_MAX).astype("<u4").tobytes(),  # modulo 2**32
                (ends - starts).astype("<u4").tobytes(),
                channels,
                lang_codes,
                token_speakers,
                _encode_confidences(segment.confidences),
                _encode_scores(segment.am_scores),
                _encode_scores(segment.lm_scores),
            )
            if self._compress is not None:
                blobs = tuple(None if b is None else self._compress(b) for b in blobs)
        return (
            recording_id,
            segment.name,
            segment.n_best_index,
            _to_ms(segment.start),
            _to_ms(segment.end),
            self._speaker_row(segment.speaker),
            num_tokens,
            *blobs,
        )

    def _language_row(self, lang_code: Optional[str]) -> int:
        """
        :return: languages.id of the lang_code, 0 for None
        """
        if lang_code is None:
            return 0
        language_id = self._language_ids.get(lang_code)
        if language_id is None:
            cursor = self._db.execute("INSERT INTO languages (lang_code) VALUES (?)", (lang_code,))
            language_id = self._language_ids[lang_code] = cursor.lastrowid
            if language_id > 0xFFFF:
                raise ValueError("more than 65535 distinct lang_codes")
        return language_id

    def _speaker_row(self, speaker: Optional[SpeakerInfo]) -> Optional[int]:
        if speaker is None:
            return None
        known = self._speaker_ids.get(speaker.id)
        if known is not None:
            if known[1] != speaker:
                raise ValueError(f"speaker {speaker.id} given with different name or gender: {known[1]}, {speaker}")
            return known[0]
        cursor = self._db.execute(
            "INSERT INTO speakers (speaker_id, name, gender) VALUES (?, ?, ?)",
            (speaker.id, speaker.name, speaker.gender),
        )
        self._speaker_ids[speaker.id] = (cursor.lastrowid, speaker)
        return cursor.lastrowid


def _encode_confidences(confidences: Optional[np.ndarray]) -> Optional[bytes]:
    if confidences is None:
        return None
    confidences = np.asarray(confidences, dtype=np.float64)
    missing = np.isnan(confidences)
    if np.any((confidences[~missing] < 0) | (confidences[~missing] > 1)):
        raise ValueError("confidences must be between 0 and 1")
    encoded = np.rint(np.where(missing, 0, confidences) * _CONFIDENCE_SCALE).astype("<u2")
    encoded[missing] = _CONFIDENCE_NONE
    return encoded.tobytes()


def _encode_scores(scores: Optional[np.ndarray]) -> Optional[bytes]:
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size and np.abs(finite).max() > _FLOAT32_MAX:
        raise ValueError("finite scores must fit into float32 (|score| <= 3.4e38)")
    return scores.astype("<f4").tobytes()


class ASRResultsDBReader:
    """
    Reads an ASR results DB::

        with ASRResultsDBReader("results.db") as db:
            for recording in db.iter_recordings():  # i6_core models
                ...
            for info in db.iter_recording_arrays():  # fast: numpy arrays, no pydantic
                ...
            recording = db.get_recording("call-0815")
    """

    def __init__(self, path: str | PathLike):
        """
        :param path: the file; it is opened read-only and treated as immutable
        """
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        uri = "file:" + urllib.parse.quote(os.path.abspath(self.path)) + "?mode=ro&immutable=1"
        self._db = sqlite3.connect(uri, uri=True)
        application_id = self._db.execute("PRAGMA application_id").fetchone()[0]
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID:
            raise ValueError(f"{self.path} is not an ASR results DB (application_id {application_id:#x})")
        if version != FORMAT_VERSION:
            raise ValueError(f"{self.path} has format version {version}, supported is {FORMAT_VERSION}")
        self.meta: dict[str, str] = dict(self._db.execute("SELECT key, value FROM meta"))
        codec = self.meta.get("codec")
        if codec == "zstd":
            self._decompress = _zstd_decompressor()
        elif codec == "none":
            self._decompress = None
        else:
            raise ValueError(f"{self.path}: unknown codec {codec!r}")
        self._audio_root = self.meta.get("audio_root")
        self._speakers = {
            row[0]: SpeakerInfo.model_construct(
                **{k: v for k, v in zip(("id", "name", "gender"), row[1:]) if v is not None}
            )
            for row in self._db.execute("SELECT id, speaker_id, name, gender FROM speakers")
        }
        self._speakers[0] = None
        self._languages: dict[int, Optional[str]] = dict(self._db.execute("SELECT id, lang_code FROM languages"))
        self._languages[0] = None

    def __enter__(self) -> "ASRResultsDBReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._db.close()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]

    def __contains__(self, name: str) -> bool:
        return self._db.execute("SELECT 1 FROM recordings WHERE name = ?", (name,)).fetchone() is not None

    def recording_names(self) -> list[str]:
        """
        :return: names of all recordings in file order
        """
        return [name for (name,) in self._db.execute("SELECT name FROM recordings ORDER BY id")]

    def execute(self, sql: str, parameters: Sequence = ()) -> sqlite3.Cursor:
        """
        Runs a read-only SQL query, e.g. to select recordings by status or duration.
        """
        return self._db.execute(sql, parameters)

    def iter_recording_arrays(self, names: Optional[Iterable[str]] = None) -> Iterator[RecordingInfo]:
        """
        :param names: recordings to read in this order, all in file order if None
        :return: the recordings with their segments as arrays
        """
        if names is not None:
            for name in names:
                yield self.get_recording_arrays(name)
            return
        # segments of different recordings may be interleaved in the file, so read them grouped by recording
        segments = self._db.execute(f"SELECT recording_id, {_SEGMENT_COLUMNS} FROM segments ORDER BY recording_id, id")
        pending = next(segments, None)
        for recording_id, *recording_row in self._db.execute(
            "SELECT id, name, path, status, error, duration_ms FROM recordings ORDER BY id"
        ):
            if pending is not None and pending[0] < recording_id:
                raise ValueError(f"{self.path}: segment {pending[1]} belongs to no recording")
            info = self._recording_info(recording_row)
            while pending is not None and pending[0] == recording_id:
                info.segments.append(self._segment(pending[1:]))
                pending = next(segments, None)
            yield info
        if pending is not None:
            raise ValueError(f"{self.path}: segment {pending[1]} belongs to no recording")

    def get_recording_arrays(self, name: str) -> RecordingInfo:
        """
        :raises KeyError: if there is no recording of that name
        """
        row = self._db.execute(
            "SELECT id, name, path, status, error, duration_ms FROM recordings WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        info = self._recording_info(row[1:])
        for segment_row in self._db.execute(
            f"SELECT {_SEGMENT_COLUMNS} FROM segments WHERE recording_id = ? ORDER BY id", (row[0],)
        ):
            info.segments.append(self._segment(segment_row))
        return info

    def iter_recordings(self, names: Optional[Iterable[str]] = None) -> Iterator[BatchRecording]:
        """
        :param names: recordings to read in this order, all in file order if None
        :return: the recordings as i6_core models
        """
        for info in self.iter_recording_arrays(names):
            yield info.to_batch_recording()

    def get_recording(self, name: str) -> BatchRecording:
        """
        :raises KeyError: if there is no recording of that name
        """
        return self.get_recording_arrays(name).to_batch_recording()

    def _recording_info(self, row: Sequence) -> RecordingInfo:
        name, path, status, error, duration_ms = row
        if path is not None and self._audio_root is not None and not os.path.isabs(path):
            path = os.path.join(self._audio_root, path)
        return RecordingInfo(name=name, path=path, status=status, error=error, duration=_from_ms(duration_ms))

    def _segment(self, row: Sequence) -> SegmentArrays:
        name, n_best_index, start_ms, end_ms, speaker, num_tokens, *blobs = row
        try:
            return self._decode_segment(name, n_best_index, start_ms, end_ms, speaker, num_tokens, blobs)
        except _FormatError as e:
            raise ValueError(f"{self.path}: segment {name}: {e}") from None

    def _decode_segment(self, name, n_best_index, start_ms, end_ms, speaker, num_tokens, blobs) -> SegmentArrays:
        if self._decompress is not None:
            blobs = [None if b is None else self._decompress(b) for b in blobs]
        tokens, start_deltas, durations, channels, lang_codes, token_speakers, confidences, am_scores, lm_scores = blobs
        if start_ms is not None and not 0 <= start_ms <= _UINT32_MAX:
            raise _FormatError(f"start {start_ms} ms out of range")
        if end_ms is not None and not 0 <= end_ms <= _UINT32_MAX:
            raise _FormatError(f"end {end_ms} ms out of range")
        if num_tokens == 0:
            if any(b is not None for b in blobs):
                raise _FormatError("per-token values for 0 tokens")
            token_list = []
            starts_ms, ends_ms = np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
        else:
            if tokens is None or start_deltas is None or durations is None:
                raise _FormatError("tokens or times missing")
            token_list = tokens.decode("utf-8").split("\0")
            if len(token_list) != num_tokens:
                raise _FormatError(f"{len(token_list)} tokens instead of {num_tokens}")
            # uint32 arithmetic wraps around at 2**32 as the spec defines; every result is a valid time
            starts_ms = _values(start_deltas, _U4, num_tokens, "start_deltas").cumsum(dtype=np.uint32)
            ends_ms = starts_ms + _values(durations, _U4, num_tokens, "durations").astype(np.int64)
        channel_array = None
        if channels is not None:
            channel_array = _values(channels, _U2, num_tokens, "channels")
            if channel_array.min() < 1:
                raise _FormatError("channel 0")
        try:
            lang_list = None
            if lang_codes is not None:
                lang_list = [self._languages[i] for i in _values(lang_codes, _U2, num_tokens, "lang_codes").tolist()]
            speaker_list = None
            if token_speakers is not None:
                indices = _values(token_speakers, _U4, num_tokens, "token_speakers").tolist()
                speaker_list = [self._speakers[i] for i in indices]
            segment_speaker = None if speaker is None else self._speakers[speaker]
        except KeyError as e:
            raise _FormatError(f"unknown lang_code or speaker index {e}") from None
        return SegmentArrays(
            name=name,
            n_best_index=n_best_index,
            start=_from_ms(start_ms),
            end=_from_ms(end_ms),
            speaker=segment_speaker,
            tokens=token_list,
            starts_ms=starts_ms,
            ends_ms=ends_ms,
            channels=channel_array,
            lang_codes=lang_list,
            token_speakers=speaker_list,
            confidences=None if confidences is None else _decode_confidences(confidences, num_tokens),
            am_scores=None if am_scores is None else _values(am_scores, _F4, num_tokens, "am_scores"),
            lm_scores=None if lm_scores is None else _values(lm_scores, _F4, num_tokens, "lm_scores"),
        )


class _FormatError(Exception):
    """
    A segment violates the format; turned into a ValueError naming the file and segment.
    """


_U2, _U4, _F4 = np.dtype("<u2"), np.dtype("<u4"), np.dtype("<f4")


def _values(data: bytes, dtype: np.dtype, num_tokens: int, what: str) -> np.ndarray:
    if len(data) != num_tokens * dtype.itemsize:
        raise _FormatError(f"{what} does not hold {num_tokens} values")
    return np.frombuffer(data, dtype=dtype)


_SEGMENT_COLUMNS = (
    "name, n_best_index, start_ms, end_ms, speaker, num_tokens, tokens, start_deltas, durations, "
    "channels, lang_codes, token_speakers, confidences, am_scores, lm_scores"
)
_RECORDING_FIELDS = ("path", "status", "error", "duration")
_PER_TOKEN_FIELDS = (
    "starts_ms",
    "ends_ms",
    "channels",
    "lang_codes",
    "token_speakers",
    "confidences",
    "am_scores",
    "lm_scores",
)


def _decode_confidences(data: bytes, num_tokens: int) -> np.ndarray:
    encoded = _values(data, _U2, num_tokens, "confidences")
    missing = encoded == _CONFIDENCE_NONE
    # every code above the scale must be the "no confidence" marker
    if np.count_nonzero(encoded > _CONFIDENCE_SCALE) != np.count_nonzero(missing):
        raise _FormatError("confidence code out of range")
    confidences = encoded / _CONFIDENCE_SCALE
    confidences[missing] = math.nan
    return confidences
