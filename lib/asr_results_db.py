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
        lang_code TEXT NOT NULL UNIQUE     -- TranscriptBase.lang_code
    );
    CREATE TABLE entities (
        id INTEGER PRIMARY KEY,            -- >= 1
        entity TEXT NOT NULL UNIQUE        -- ReferenceTranscript.entity
    );
    CREATE TABLE recordings (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,         -- Recording.name
        path TEXT,                         -- Recording.path; with meta "audio_root": relative to it, or absolute
        status TEXT NOT NULL DEFAULT 'OK', -- e.g. 'OK', 'ASR_ERROR'; not part of the i6_core models
        error TEXT,                        -- error message if status is not 'OK'
        duration_ms INTEGER                -- Recording.duration
    );
    -- segments of ASR hypotheses (TimedTranscript)
    CREATE TABLE hypothesis_segments (
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
    -- segments of evaluation references (ReferenceTranscript)
    CREATE TABLE reference_segments (
        id INTEGER PRIMARY KEY,            -- ascending in the order of the segments in the recording
        recording_id INTEGER NOT NULL REFERENCES recordings(id),
        name TEXT NOT NULL,                -- Segment.name
        start_ms INTEGER,                  -- Segment.start
        end_ms INTEGER,                    -- Segment.end
        speaker INTEGER REFERENCES speakers(id),  -- Segment.speaker
        num_tokens INTEGER NOT NULL,       -- number of ReferenceTranscripts of the segment
        tokens BLOB,                       -- ReferenceTranscript.text
        start_deltas BLOB,                 -- ReferenceTranscript.start
        durations BLOB,                    -- ReferenceTranscript.end - ReferenceTranscript.start
        timed BLOB,                        -- whether the ReferenceTranscript has start and end
        channels BLOB,                     -- ReferenceTranscript.channel
        lang_codes BLOB,                   -- ReferenceTranscript.lang_code
        token_speakers BLOB,               -- ReferenceTranscript.speaker
        entities BLOB,                     -- ReferenceTranscript.entity
        entity_confidences BLOB,           -- ReferenceTranscript.entity_confidence
        CHECK ((start_ms IS NULL) = (end_ms IS NULL))
    );
    -- properties of speakers, recordings and segments, in the order given
    CREATE TABLE speaker_properties (
        speaker_id INTEGER NOT NULL REFERENCES speakers(id),
        key TEXT NOT NULL,
        kind TEXT NOT NULL,                -- 'str', 'int', 'float' or 'bool': the Python type of the value
        value NOT NULL,                    -- TEXT for 'str', INTEGER for 'int' and 'bool' (0 or 1), REAL for 'float'
        UNIQUE (speaker_id, key)
    );
    CREATE TABLE recording_properties (recording_id ... REFERENCES recordings(id), key, kind, value, UNIQUE ...);
    CREATE TABLE hypothesis_segment_properties (segment_id ... REFERENCES hypothesis_segments(id), ...);
    CREATE TABLE reference_segment_properties (segment_id ... REFERENCES reference_segments(id), ...);
    CREATE INDEX hypothesis_segments_recording ON hypothesis_segments (recording_id, id);
    CREATE INDEX reference_segments_recording ON reference_segments (recording_id, id);

(``_SCHEMA`` below has the property tables in full.)

Rules:

* Every field of the i6_core models is stored where the model has it: Segment fields as columns of the segment
  tables, transcript fields per token, properties in the property tables.
* A recording holds either hypothesis or reference segments, not both, like a JSONL file holds one kind of
  transcript. A recording without segments may be read as either.
* All times are integer milliseconds from the beginning of the recording: ``seconds * 1000`` computed in double
  precision and rounded half to even (C ``rint`` in the default rounding mode, ``numpy.rint``, Python ``round``).
* The segments of a recording are its rows of the segment table in ``id`` order. Rows of different recordings may
  be interleaved, e.g. when results of several recordings are written as they come in; readers group them by
  ``recording_id``. Every segment must belong to a recording.
* Segments of a recording with the same name next to each other are the n-best alternatives of one segment (as in
  the JSONL format), ordered by ``n_best_index`` (None first) when read as i6_core models. Empty groups, groups
  mixing names and neighbouring groups of the same name cannot be represented; writers reject them. Reference
  segments have no ``n_best_index`` and either both or neither of start and end.
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
    Hypothesis tokens always have times. For reference segments, ``start_deltas`` and ``durations`` are NULL if no
    token has times; tokens without times are stored with start and end 0.
  - ``timed`` (references): uint8, 1 if the token has start and end, 0 if not. NULL if all tokens have times or none
    has (then ``start_deltas`` and ``durations`` are NULL).
  - ``channels``: little-endian uint16, the 1-based channel. NULL if all tokens are on channel 1.
  - ``lang_codes``: little-endian uint16, ``languages.id`` of the token's lang_code, 0 if it has none. NULL if no
    token has one.
  - ``token_speakers``: little-endian uint32, ``speakers.id`` of the token's speaker, 0 if it has none. NULL if no
    token has one.
  - ``confidences`` (hypotheses), ``entity_confidences`` (references): little-endian uint16 holding
    ``confidence * 10000`` rounded half to even (0..10000), 65535 if the token has none. NULL if no token has one.
  - ``am_scores``, ``lm_scores`` (hypotheses): little-endian float32, NaN if the token has no score. NULL if no token
    has one.
  - ``entities`` (references): little-endian uint32, ``entities.id`` of the token's entity, 0 if it has none. NULL if
    no token has one. A token with an entity_confidence has an entity.

  A blob that is NULL for "all default" may also be written with the default values; readers must accept both.

* Properties: one row per key and value. ``kind`` gives the Python type the value is read back as, so that e.g. 30,
  30.0 and True stay distinct. Float values are finite. An empty properties dict is not stored and reads back as
  None.
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
  - integer property values -2**63 .. 2**63 - 1;
  - texts must be valid UTF-8 (no lone surrogates) and must not contain NUL.

* Readers must reject as format errors: per-token blobs whose number of values differs from ``num_tokens``, segment
  ``start_ms`` / ``end_ms`` outside the limits, confidence codes outside 0..10000 and 65535, channels 0, ``timed``
  values other than 0 and 1, lang_code, speaker or entity indices that do not exist, entity confidences without an
  entity, and property kinds other than the four above or not matching the value. Token ends above the limit (up to
  ``2 * (2**32 - 1)``, which a writer never produces) need not be rejected.
* Files are written once and then only read. Readers may open them with ``?immutable=1``.
"""

__all__ = [
    "FORMAT_VERSION",
    "APPLICATION_ID",
    "HypothesisSegmentArrays",
    "ReferenceSegmentArrays",
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
from typing import Any, Optional, Union

import numpy as np

from i6_core.lib.asr_results import (
    BatchRecording,
    BatchSegment,
    ReferenceTranscript,
    SpeakerInfo,
    TimedTranscript,
    TranscriptBase,
)

FORMAT_VERSION = 1
APPLICATION_ID = 0x41535244  # "ASRD"

_CONFIDENCE_SCALE = 10000
_CONFIDENCE_NONE = 0xFFFF
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _property_table(name: str, owner: str, owner_table: str) -> str:
    return f"""
CREATE TABLE {name} (
    {owner} INTEGER NOT NULL REFERENCES {owner_table}(id),
    key TEXT NOT NULL,
    kind TEXT NOT NULL,
    value NOT NULL,
    UNIQUE ({owner}, key)
);"""


_SCHEMA = (
    """
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
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    entity TEXT NOT NULL UNIQUE
);
CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT,
    status TEXT NOT NULL DEFAULT 'OK',
    error TEXT,
    duration_ms INTEGER
);
CREATE TABLE hypothesis_segments (
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
CREATE TABLE reference_segments (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    name TEXT NOT NULL,
    start_ms INTEGER,
    end_ms INTEGER,
    speaker INTEGER REFERENCES speakers(id),
    num_tokens INTEGER NOT NULL,
    tokens BLOB,
    start_deltas BLOB,
    durations BLOB,
    timed BLOB,
    channels BLOB,
    lang_codes BLOB,
    token_speakers BLOB,
    entities BLOB,
    entity_confidences BLOB,
    CHECK ((start_ms IS NULL) = (end_ms IS NULL))
);"""
    + _property_table("speaker_properties", "speaker_id", "speakers")
    + _property_table("recording_properties", "recording_id", "recordings")
    + _property_table("hypothesis_segment_properties", "segment_id", "hypothesis_segments")
    + _property_table("reference_segment_properties", "segment_id", "reference_segments")
    + """
CREATE INDEX hypothesis_segments_recording ON hypothesis_segments (recording_id, id);
CREATE INDEX reference_segments_recording ON reference_segments (recording_id, id);
"""
)


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


def _optional_list(values: list) -> Optional[list]:
    return None if all(v is None for v in values) else values


def _optional_array(values: Sequence[Optional[float]], dtype) -> Optional[np.ndarray]:
    if all(v is None for v in values):
        return None
    return np.array([math.nan if v is None else v for v in values], dtype=dtype)


def _common_token_values(transcripts: Sequence[TranscriptBase]) -> dict[str, Any]:
    """
    :return: the fields of TranscriptBase as HypothesisSegmentArrays / ReferenceSegmentArrays fields
    """
    channels = [t.channel for t in transcripts]
    return {
        "tokens": [t.text for t in transcripts],
        "channels": None if all(c == 1 for c in channels) else np.array(channels, dtype=np.int64),
        "lang_codes": _optional_list([t.lang_code for t in transcripts]),
        "token_speakers": _optional_list([t.speaker for t in transcripts]),
    }


def _segment_values(segment: BatchSegment) -> dict[str, Any]:
    """
    :return: the Segment fields as HypothesisSegmentArrays / ReferenceSegmentArrays fields
    """
    return {
        "name": segment.name,
        "start": segment.start,
        "end": segment.end,
        "speaker": segment.speaker,
        "properties": segment.properties,
    }


def _token_dicts(
    segment: Union["HypothesisSegmentArrays", "ReferenceSegmentArrays"], type_: str
) -> list[dict[str, Any]]:
    """
    :return: per token the dict of its TranscriptBase fields, unset ones omitted, with the `type` discriminator
    """
    channels = None if segment.channels is None else segment.channels.tolist()
    dicts = []
    for i, text in enumerate(segment.tokens):
        values = {"type": type_, "text": text}
        # the default channel is left unset, so that JSONL written with exclude_unset omits it as for the original
        if channels is not None and channels[i] != 1:
            values["channel"] = channels[i]
        if segment.lang_codes is not None and segment.lang_codes[i] is not None:
            values["lang_code"] = segment.lang_codes[i]
        if segment.token_speakers is not None and segment.token_speakers[i] is not None:
            values["speaker"] = segment.token_speakers[i]
        dicts.append(values)
    return dicts


def _batch_segment(
    segment: Union["HypothesisSegmentArrays", "ReferenceSegmentArrays"], transcripts: list[dict]
) -> BatchSegment:
    values = {"name": segment.name, "transcripts": transcripts}
    for key in ("n_best_index", "start", "end", "speaker", "properties"):
        if getattr(segment, key, None) is not None:
            values[key] = getattr(segment, key)
    # validating the whole segment at once is ~3x faster than model_construct per transcript
    return BatchSegment.model_validate(values)


@dataclass
class HypothesisSegmentArrays:
    """
    One hypothesis segment with its tokens (TimedTranscripts) as arrays: the unit in which segments are stored, and
    the fast way to read them.

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
    properties: Optional[dict[str, Any]] = None

    @classmethod
    def from_batch_segment(cls, segment: BatchSegment) -> "HypothesisSegmentArrays":
        transcripts = segment.transcripts
        return cls(
            **_segment_values(segment),
            **_common_token_values(transcripts),
            n_best_index=segment.n_best_index,
            starts_ms=np.array([_to_ms(t.start) for t in transcripts], dtype=np.int64),
            ends_ms=np.array([_to_ms(t.end) for t in transcripts], dtype=np.int64),
            confidences=_optional_array([t.confidence for t in transcripts], np.float64),
            am_scores=_optional_array([t.am_score for t in transcripts], np.float32),
            lm_scores=_optional_array([t.lm_score for t in transcripts], np.float32),
        )

    def to_batch_segment(self) -> BatchSegment:
        """
        :return: the segment as i6_core model with TimedTranscripts; fields that are None are left unset
        """
        transcripts = _token_dicts(self, "tt")
        starts, ends = self.starts_ms.tolist(), self.ends_ms.tolist()
        scores = [
            (key, array.tolist())
            for key, array in (
                ("confidence", self.confidences),
                ("am_score", self.am_scores),
                ("lm_score", self.lm_scores),
            )
            if array is not None
        ]
        for i, values in enumerate(transcripts):
            values["start"], values["end"] = starts[i] / 1000, ends[i] / 1000
            for key, array in scores:
                if not math.isnan(array[i]):
                    values[key] = array[i]
        return _batch_segment(self, transcripts)


@dataclass
class ReferenceSegmentArrays:
    """
    One reference segment with its tokens (ReferenceTranscripts) as arrays.

    Times are in seconds for the segment and in integer milliseconds for the tokens. ``starts_ms`` and ``ends_ms``
    are None if no token has times; ``timed`` is None if all tokens have times (or none has), else it says per token
    whether it has, the times of tokens without being ignored (0 when read). The per-token fields after ``timed`` are
    None if all tokens have the default (channel 1, no language/speaker/entity/entity confidence); otherwise they hold
    one value per token, missing entity confidences being NaN.
    """

    name: str
    tokens: list[str]
    starts_ms: Optional[np.ndarray] = None  # uint32 when read, None = no token has times
    ends_ms: Optional[np.ndarray] = None  # int64 when read, None = no token has times
    timed: Optional[np.ndarray] = None  # bool, None = all tokens have times, or none has
    channels: Optional[np.ndarray] = None  # 1-based, None = all 1
    lang_codes: Optional[list[Optional[str]]] = None
    token_speakers: Optional[list[Optional[SpeakerInfo]]] = None
    entities: Optional[list[Optional[str]]] = None
    entity_confidences: Optional[np.ndarray] = None  # float, NaN = none
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[SpeakerInfo] = None
    properties: Optional[dict[str, Any]] = None

    @classmethod
    def from_batch_segment(cls, segment: BatchSegment) -> "ReferenceSegmentArrays":
        if segment.n_best_index is not None:
            raise ValueError(f"reference segment {segment.name} has an n_best_index")
        transcripts = segment.transcripts
        timed = [t.start is not None for t in transcripts]
        starts_ms = ends_ms = timed_array = None
        if any(timed):
            starts_ms = np.array([_to_ms(t.start) or 0 for t in transcripts], dtype=np.int64)
            ends_ms = np.array([_to_ms(t.end) or 0 for t in transcripts], dtype=np.int64)
            if not all(timed):
                timed_array = np.array(timed, dtype=bool)
        return cls(
            **_segment_values(segment),
            **_common_token_values(transcripts),
            starts_ms=starts_ms,
            ends_ms=ends_ms,
            timed=timed_array,
            entities=_optional_list([t.entity for t in transcripts]),
            entity_confidences=_optional_array([t.entity_confidence for t in transcripts], np.float64),
        )

    def to_batch_segment(self) -> BatchSegment:
        """
        :return: the segment as i6_core model with ReferenceTranscripts; fields that are None are left unset
        """
        transcripts = _token_dicts(self, "rt")
        if self.starts_ms is not None:
            starts, ends = self.starts_ms.tolist(), self.ends_ms.tolist()
            timed = [True] * len(starts) if self.timed is None else self.timed.tolist()
            for i, values in enumerate(transcripts):
                if timed[i]:
                    values["start"], values["end"] = starts[i] / 1000, ends[i] / 1000
        entity_confidences = None if self.entity_confidences is None else self.entity_confidences.tolist()
        for i, values in enumerate(transcripts):
            if self.entities is not None and self.entities[i] is not None:
                values["entity"] = self.entities[i]
            if entity_confidences is not None and not math.isnan(entity_confidences[i]):
                values["entity_confidence"] = entity_confidences[i]
        return _batch_segment(self, transcripts)


@dataclass
class RecordingInfo:
    """
    A recording with its segments as arrays, and the fields of this format that the i6_core models do not have.
    """

    name: str
    path: Optional[str] = None
    status: str = "OK"
    error: Optional[str] = None
    duration: Optional[float] = None  # seconds
    properties: Optional[dict[str, Any]] = None
    segments: list[Union[HypothesisSegmentArrays, ReferenceSegmentArrays]] = field(default_factory=list)

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
        for key in ("path", "duration", "properties"):
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        # validate the recording fields (path becomes the same type as when parsing JSONL), not the segments again
        recording = BatchRecording.model_validate(values)
        recording.alternatives = alternatives
        return recording


@dataclass
class _RecordingState:
    """
    What the writer keeps of a recording written so far, to continue it.
    """

    id: int
    path: Optional[str]
    status: str
    error: Optional[str]
    duration_ms: Optional[int]
    properties: Optional[dict[str, Any]]
    kind: Optional[type] = None  # HypothesisSegmentArrays or ReferenceSegmentArrays, None while it has no segments
    last_segment: Optional[str] = None


_ARRAYS_OF_TRANSCRIPT_TYPE = {TimedTranscript: HypothesisSegmentArrays, ReferenceTranscript: ReferenceSegmentArrays}


def _arrays_type(transcript_type: type[TranscriptBase]) -> type:
    arrays_type = _ARRAYS_OF_TRANSCRIPT_TYPE.get(transcript_type)
    if arrays_type is None:
        raise TypeError(f"transcript_type must be TimedTranscript or ReferenceTranscript, got {transcript_type!r}")
    return arrays_type


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
        self._recordings: dict[str, _RecordingState] = {}
        self._speaker_ids: dict[str, tuple[int, SpeakerInfo]] = {}
        self._language_ids: dict[str, int] = {}
        self._entity_ids: dict[str, int] = {}

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

    def add_recording(
        self,
        recording: BatchRecording,
        transcript_type: type[TranscriptBase] = TimedTranscript,
        *,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        :param recording: the recording with all its segments
        :param transcript_type: kind of transcript of the recording, TimedTranscript (ASR hypotheses) or
            ReferenceTranscript, as for :func:`i6_core.lib.asr_results.write_asr_results`; it decides where segments
            without transcripts go
        :param status: see :meth:`add_recording_arrays`
        :param error: see :meth:`add_recording_arrays`
        :raises TypeError: for a transcript of another type
        :raises ValueError: for n-best structures that a flat sequence of segments cannot represent: an empty
            group, a group mixing segment names, or two neighbouring groups with the same segment name
        """
        arrays_type = _arrays_type(transcript_type)
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
            for segment in alternative:
                for transcript in segment.transcripts:
                    if type(transcript) is not transcript_type:
                        raise TypeError(
                            f"recording {recording.name}: expected {transcript_type.__name__}, "
                            f"got {type(transcript).__name__}"
                        )
                segments.append(arrays_type.from_batch_segment(segment))
        path = None if recording.path is None else os.fspath(recording.path)
        self.add_recording_arrays(
            recording.name,
            segments,
            path=path,
            status=status,
            error=error,
            duration=recording.duration,
            properties=recording.properties,
        )

    def add_recording_arrays(
        self,
        name: str,
        segments: Iterable[Union[HypothesisSegmentArrays, ReferenceSegmentArrays]],
        *,
        path: Optional[str] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None,
        properties: Optional[Mapping[str, Any]] = None,
    ):
        """
        Adds a recording given as arrays, without going through the i6_core models.

        A recording may be added in several calls, e.g. to write the results of several recordings as they come in:
        a call with the name of a recording that was already added appends the segments to it. The other arguments
        of such a call must be None or equal to the stored values.

        :param name: name of the recording
        :param segments: the segments in recording order, n-best alternatives of a segment next to each other; all
            HypothesisSegmentArrays (hypotheses) or all ReferenceSegmentArrays, also over several calls
        :param path: audio file of the recording
        :param status: what went wrong; None for "OK"
        :param error: error message for a status other than "OK"
        :param duration: duration of the audio in seconds
        :param properties: Recording.properties
        """
        segments = list(segments)
        kinds = {type(segment) for segment in segments}
        if not kinds <= {HypothesisSegmentArrays, ReferenceSegmentArrays}:
            raise TypeError(f"recording {name}: segments must be HypothesisSegmentArrays or ReferenceSegmentArrays")
        if len(kinds) > 1:
            raise ValueError(f"recording {name} has both hypothesis and reference segments")
        kind = next(iter(kinds), None)
        stored_path = self._stored_path(path)
        duration_ms = _to_ms(duration)
        properties = None if properties is None else dict(properties)
        property_rows = _property_rows(properties, f"recording {name}")
        # all or nothing: a rejected call leaves no rows behind, also no speakers, languages or entities it introduced
        caches = dict(self._speaker_ids), dict(self._language_ids), dict(self._entity_ids)
        self._db.execute("SAVEPOINT recording")
        try:
            state = self._recordings.get(name)
            if state is None:
                cursor = self._db.execute(
                    "INSERT INTO recordings (name, path, status, error, duration_ms) VALUES (?, ?, ?, ?, ?)",
                    (name, stored_path, status or "OK", error, duration_ms),
                )
                state = _RecordingState(cursor.lastrowid, stored_path, status or "OK", error, duration_ms, properties)
                self._insert_properties("recording_properties", "recording_id", state.id, property_rows)
                is_new = True
            else:
                is_new = False
                given = (stored_path, status, error, duration_ms, properties)
                stored = (state.path, state.status, state.error, state.duration_ms, state.properties)
                for given_value, value, what in zip(given, stored, _RECORDING_FIELDS):
                    if given_value is not None and given_value != value:
                        raise ValueError(
                            f"recording {name} added again with a different {what}: {given_value!r}, {value!r}"
                        )
                if kind is not None and state.kind is not None and kind is not state.kind:
                    raise ValueError(f"recording {name} has both hypothesis and reference segments")
                if segments and segments[0].name == state.last_segment:
                    raise ValueError(
                        f"recording {name} continued with segment {state.last_segment}, which would join the n-best "
                        "alternatives written before"
                    )
            for segment in segments:
                if kind is HypothesisSegmentArrays:
                    segment_id = self._db.execute(_INSERT_HYPOTHESIS, self._hypothesis_row(state.id, segment)).lastrowid
                    properties_table = "hypothesis_segment_properties"
                else:
                    segment_id = self._db.execute(_INSERT_REFERENCE, self._reference_row(state.id, segment)).lastrowid
                    properties_table = "reference_segment_properties"
                rows = _property_rows(segment.properties, f"segment {segment.name}")
                self._insert_properties(properties_table, "segment_id", segment_id, rows)
        except BaseException:
            self._db.execute("ROLLBACK TO recording")
            self._db.execute("RELEASE recording")
            self._speaker_ids, self._language_ids, self._entity_ids = caches
            raise
        self._db.execute("RELEASE recording")
        if is_new:
            self._recordings[name] = state
        if segments:
            state.kind = kind
            state.last_segment = segments[-1].name

    def _stored_path(self, path: Optional[str]) -> Optional[str]:
        """
        :return: the path as stored: unchanged without audio_root, else absolute or relative to audio_root
        """
        if path is None or self._audio_root is None:
            return path
        path = os.path.abspath(path)
        prefix = self._audio_root.rstrip(os.sep) + os.sep
        return path[len(prefix) :] if path.startswith(prefix) else path

    def _insert_properties(self, table: str, owner: str, owner_id: int, rows: list[tuple]):
        if rows:
            self._db.executemany(
                f"INSERT INTO {table} ({owner}, key, kind, value) VALUES (?, ?, ?, ?)",
                ((owner_id, *row) for row in rows),
            )

    def _check_segment(self, segment: Union[HypothesisSegmentArrays, ReferenceSegmentArrays], fields: Sequence[str]):
        num_tokens = len(segment.tokens)
        for name in fields:
            values = getattr(segment, name)
            if values is not None and len(values) != num_tokens:
                raise ValueError(f"segment {segment.name}: {name} has {len(values)} values for {num_tokens} tokens")
        start, end = _to_ms(segment.start), _to_ms(segment.end)
        if start is not None and end is not None and start > end:
            raise ValueError(f"segment {segment.name}: start > end")
        return start, end

    def _tokens_blob(self, segment: Union[HypothesisSegmentArrays, ReferenceSegmentArrays]) -> bytes:
        if any("\0" in token for token in segment.tokens):
            raise ValueError(f"segment {segment.name}: tokens must not contain NUL characters")
        try:
            return "\0".join(segment.tokens).encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(f"segment {segment.name}: tokens are not valid unicode: {e}") from None

    @staticmethod
    def _time_blobs(name: str, starts_ms, ends_ms, timed: Optional[np.ndarray] = None) -> tuple[bytes, bytes]:
        starts = np.asarray(starts_ms, dtype=np.int64)
        ends = np.asarray(ends_ms, dtype=np.int64)
        if timed is not None:
            # the times of tokens without are not stored
            starts, ends = np.where(timed, starts, 0), np.where(timed, ends, 0)
        if starts.min() < 0 or ends.max() > _UINT32_MAX or np.any(starts > ends):
            raise ValueError(f"segment {name}: token times out of range or start > end")
        return (
            (np.diff(starts, prepend=0) & _UINT32_MAX).astype("<u4").tobytes(),  # modulo 2**32
            (ends - starts).astype("<u4").tobytes(),
        )

    def _common_blobs(self, segment: Union[HypothesisSegmentArrays, ReferenceSegmentArrays]) -> tuple:
        """
        :return: the channels, lang_codes and token_speakers blobs
        """
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
        return channels, lang_codes, token_speakers

    def _compressed(self, blobs: tuple) -> tuple:
        if self._compress is None:
            return blobs
        return tuple(None if b is None else self._compress(b) for b in blobs)

    def _hypothesis_row(self, recording_id: int, segment: HypothesisSegmentArrays) -> tuple:
        start, end = self._check_segment(segment, _HYPOTHESIS_PER_TOKEN_FIELDS)
        if segment.n_best_index is not None and not 0 <= segment.n_best_index < 2**63:
            raise ValueError(f"segment {segment.name}: n_best_index {segment.n_best_index} out of range")
        if not segment.tokens:
            blobs = (None,) * 9
        else:
            blobs = self._compressed(
                (
                    self._tokens_blob(segment),
                    *self._time_blobs(segment.name, segment.starts_ms, segment.ends_ms),
                    *self._common_blobs(segment),
                    _encode_confidences(segment.confidences, segment.name, "confidences"),
                    _encode_scores(segment.am_scores),
                    _encode_scores(segment.lm_scores),
                )
            )
        return (
            recording_id,
            segment.name,
            segment.n_best_index,
            start,
            end,
            self._speaker_row(segment.speaker),
            len(segment.tokens),
            *blobs,
        )

    def _reference_row(self, recording_id: int, segment: ReferenceSegmentArrays) -> tuple:
        start, end = self._check_segment(segment, _REFERENCE_PER_TOKEN_FIELDS)
        if (start is None) != (end is None):
            raise ValueError(f"reference segment {segment.name} must have both start and end time, or neither")
        if (segment.starts_ms is None) != (segment.ends_ms is None):
            raise ValueError(f"segment {segment.name}: starts_ms and ends_ms must both be given or both be None")
        if segment.timed is not None and segment.starts_ms is None:
            raise ValueError(f"segment {segment.name}: timed is given without token times")
        if not segment.tokens:
            blobs = (None,) * 9
        else:
            start_deltas = durations = timed = None
            if segment.starts_ms is not None:
                timed_array = None if segment.timed is None else np.asarray(segment.timed, dtype=bool)
                if timed_array is None or timed_array.all():
                    start_deltas, durations = self._time_blobs(segment.name, segment.starts_ms, segment.ends_ms)
                elif timed_array.any():
                    start_deltas, durations = self._time_blobs(
                        segment.name, segment.starts_ms, segment.ends_ms, timed_array
                    )
                    timed = timed_array.astype("u1").tobytes()
            entities = None
            if segment.entities is not None:
                entities = np.array([self._entity_row(e) for e in segment.entities], dtype="<u4").tobytes()
            if segment.entity_confidences is not None:
                has_entity = (
                    np.zeros(len(segment.tokens), bool)
                    if segment.entities is None
                    else np.array([e is not None for e in segment.entities])
                )
                if np.any(~np.isnan(np.asarray(segment.entity_confidences, dtype=np.float64)) & ~has_entity):
                    raise ValueError(f"segment {segment.name}: entity_confidence without entity")
            blobs = self._compressed(
                (
                    self._tokens_blob(segment),
                    start_deltas,
                    durations,
                    timed,
                    *self._common_blobs(segment),
                    entities,
                    _encode_confidences(segment.entity_confidences, segment.name, "entity_confidences"),
                )
            )
        return (recording_id, segment.name, start, end, self._speaker_row(segment.speaker), len(segment.tokens), *blobs)

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

    def _entity_row(self, entity: Optional[str]) -> int:
        """
        :return: entities.id of the entity, 0 for None
        """
        if entity is None:
            return 0
        entity_id = self._entity_ids.get(entity)
        if entity_id is None:
            cursor = self._db.execute("INSERT INTO entities (entity) VALUES (?)", (entity,))
            entity_id = self._entity_ids[entity] = cursor.lastrowid
        return entity_id

    def _speaker_row(self, speaker: Optional[SpeakerInfo]) -> Optional[int]:
        if speaker is None:
            return None
        known = self._speaker_ids.get(speaker.id)
        if known is not None:
            if known[1] != speaker:
                raise ValueError(
                    f"speaker {speaker.id} given with different name, gender or properties: {known[1]}, {speaker}"
                )
            return known[0]
        rows = _property_rows(speaker.properties, f"speaker {speaker.id}")
        cursor = self._db.execute(
            "INSERT INTO speakers (speaker_id, name, gender) VALUES (?, ?, ?)",
            (speaker.id, speaker.name, speaker.gender),
        )
        self._insert_properties("speaker_properties", "speaker_id", cursor.lastrowid, rows)
        self._speaker_ids[speaker.id] = (cursor.lastrowid, speaker)
        return cursor.lastrowid


_INSERT_HYPOTHESIS = (
    "INSERT INTO hypothesis_segments (recording_id, name, n_best_index, start_ms, end_ms, speaker, num_tokens, "
    "tokens, start_deltas, durations, channels, lang_codes, token_speakers, confidences, am_scores, lm_scores) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_REFERENCE = (
    "INSERT INTO reference_segments (recording_id, name, start_ms, end_ms, speaker, num_tokens, tokens, "
    "start_deltas, durations, timed, channels, lang_codes, token_speakers, entities, entity_confidences) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _property_rows(properties: Optional[Mapping[str, Any]], owner: str) -> list[tuple[str, str, Any]]:
    """
    :return: (key, kind, value) per property, in the given order
    """
    if not properties:
        return []
    rows = []
    for key, value in properties.items():
        if not isinstance(key, str):
            raise ValueError(f"{owner}: property key {key!r} is not a string")
        # bool before int: bool is a subclass of int
        if isinstance(value, bool):
            rows.append((key, "bool", int(value)))
        elif isinstance(value, int):
            if not -(2**63) <= value < 2**63:
                raise ValueError(f"{owner}: property {key} = {value} does not fit into 64 bits")
            rows.append((key, "int", value))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{owner}: property {key} = {value} is not finite")
            rows.append((key, "float", value))
        elif isinstance(value, str):
            rows.append((key, "str", value))
        else:
            raise ValueError(f"{owner}: property {key} has a value of type {type(value).__name__}")
    return rows


def _encode_confidences(confidences: Optional[np.ndarray], name: str, what: str) -> Optional[bytes]:
    if confidences is None:
        return None
    confidences = np.asarray(confidences, dtype=np.float64)
    missing = np.isnan(confidences)
    if np.any((confidences[~missing] < 0) | (confidences[~missing] > 1)):
        raise ValueError(f"segment {name}: {what} must be between 0 and 1")
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

    A file may hold recordings of hypotheses and of references. The reading methods take a ``transcript_type``
    (TimedTranscript or ReferenceTranscript) and then return the recordings of that kind, including those without
    segments. Without it, they return the kind the file holds, and require it for a file holding both.
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
        speaker_properties = self._properties("speaker_properties", "speaker_id")
        self._speakers: dict[int, Optional[SpeakerInfo]] = {}
        for speaker_id, *values in self._db.execute("SELECT id, speaker_id, name, gender FROM speakers"):
            fields = {k: v for k, v in zip(("id", "name", "gender"), values) if v is not None}
            if speaker_id in speaker_properties:
                fields["properties"] = speaker_properties[speaker_id]
            self._speakers[speaker_id] = SpeakerInfo.model_construct(**fields)
        self._speakers[0] = None
        self._languages: dict[int, Optional[str]] = dict(self._db.execute("SELECT id, lang_code FROM languages"))
        self._languages[0] = None
        self._entities: dict[int, Optional[str]] = dict(self._db.execute("SELECT id, entity FROM entities"))
        self._entities[0] = None
        # {table: {segment id: properties}}, loaded when first needed
        self._segment_properties: dict[str, dict[int, dict[str, Any]]] = {}

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

    @property
    def transcript_types(self) -> list[type[TranscriptBase]]:
        """
        :return: the kinds of transcript the file has segments of: TimedTranscript and/or ReferenceTranscript
        """
        return [
            transcript_type
            for transcript_type, table in _SEGMENT_TABLES.items()
            if self._db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
        ]

    def recording_names(self) -> list[str]:
        """
        :return: names of all recordings in file order, of both kinds
        """
        return [name for (name,) in self._db.execute("SELECT name FROM recordings ORDER BY id")]

    def execute(self, sql: str, parameters: Sequence = ()) -> sqlite3.Cursor:
        """
        Runs a read-only SQL query, e.g. to select recordings by status or duration.
        """
        return self._db.execute(sql, parameters)

    def iter_recording_arrays(
        self,
        names: Optional[Iterable[str]] = None,
        transcript_type: Optional[type[TranscriptBase]] = None,
    ) -> Iterator[RecordingInfo]:
        """
        :param names: recordings to read in this order, all of the kind in file order if None
        :param transcript_type: TimedTranscript or ReferenceTranscript, see the class docstring
        :return: the recordings with their segments as arrays
        """
        if names is not None:
            for name in names:
                yield self.get_recording_arrays(name, transcript_type)
            return
        transcript_type = self._file_transcript_type(transcript_type)
        table = _SEGMENT_TABLES[transcript_type]
        (other_table,) = (t for t in _SEGMENT_TABLES.values() if t != table)
        other_recordings = {i for (i,) in self._db.execute(f"SELECT DISTINCT recording_id FROM {other_table}")}
        recording_properties = self._properties("recording_properties", "recording_id")
        # segments of different recordings may be interleaved in the file, so read them grouped by recording
        segments = self._db.execute(
            f"SELECT recording_id, id, {_SEGMENT_COLUMNS[table]} FROM {table} ORDER BY recording_id, id"
        )
        pending = next(segments, None)
        for recording_id, *recording_row in self._db.execute(
            "SELECT id, name, path, status, error, duration_ms FROM recordings ORDER BY id"
        ):
            if pending is not None and pending[0] < recording_id:
                raise ValueError(f"{self.path}: segment {pending[2]} belongs to no recording")
            if recording_id in other_recordings:
                if pending is not None and pending[0] == recording_id:
                    raise ValueError(f"{self.path}: recording {recording_row[0]} has hypothesis and reference segments")
                continue
            info = self._recording_info(recording_row, recording_properties.get(recording_id))
            while pending is not None and pending[0] == recording_id:
                info.segments.append(self._segment(table, pending[1], pending[2:]))
                pending = next(segments, None)
            yield info
        if pending is not None:
            raise ValueError(f"{self.path}: segment {pending[2]} belongs to no recording")

    def get_recording_arrays(self, name: str, transcript_type: Optional[type[TranscriptBase]] = None) -> RecordingInfo:
        """
        :param transcript_type: TimedTranscript or ReferenceTranscript; None for the kind of the recording
        :raises KeyError: if there is no recording of that name
        :raises ValueError: if the recording has segments of the other kind than `transcript_type`
        """
        row = self._db.execute(
            "SELECT id, name, path, status, error, duration_ms FROM recordings WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        recording_id = row[0]
        kinds = [
            t
            for t, table in _SEGMENT_TABLES.items()
            if self._db.execute(f"SELECT 1 FROM {table} WHERE recording_id = ? LIMIT 1", (recording_id,)).fetchone()
        ]
        if transcript_type is None:
            transcript_type = kinds[0] if kinds else TimedTranscript
        else:
            _arrays_type(transcript_type)
            if kinds and kinds != [transcript_type]:
                raise ValueError(f"{self.path}: recording {name} has no {transcript_type.__name__} segments")
        properties = self._properties("recording_properties", "recording_id", recording_id).get(recording_id)
        info = self._recording_info(row[1:], properties)
        table = _SEGMENT_TABLES[transcript_type]
        for segment_id, *segment_row in self._db.execute(
            f"SELECT id, {_SEGMENT_COLUMNS[table]} FROM {table} WHERE recording_id = ? ORDER BY id", (recording_id,)
        ):
            info.segments.append(self._segment(table, segment_id, segment_row))
        return info

    def iter_recordings(
        self,
        names: Optional[Iterable[str]] = None,
        transcript_type: Optional[type[TranscriptBase]] = None,
    ) -> Iterator[BatchRecording]:
        """
        :param names: recordings to read in this order, all of the kind in file order if None
        :param transcript_type: TimedTranscript or ReferenceTranscript, see the class docstring
        :return: the recordings as i6_core models
        """
        for info in self.iter_recording_arrays(names, transcript_type):
            yield info.to_batch_recording()

    def get_recording(self, name: str, transcript_type: Optional[type[TranscriptBase]] = None) -> BatchRecording:
        """
        :raises KeyError: if there is no recording of that name
        """
        return self.get_recording_arrays(name, transcript_type).to_batch_recording()

    def _file_transcript_type(self, transcript_type: Optional[type[TranscriptBase]]) -> type[TranscriptBase]:
        if transcript_type is not None:
            _arrays_type(transcript_type)
            return transcript_type
        present = self.transcript_types
        if len(present) > 1:
            raise ValueError(
                f"{self.path} has hypothesis and reference segments, give transcript_type "
                "(TimedTranscript or ReferenceTranscript)"
            )
        return present[0] if present else TimedTranscript

    def _properties(self, table: str, owner: str, owner_id: Optional[int] = None) -> dict[int, dict[str, Any]]:
        """
        :return: {owner id: properties}, of all owners or of `owner_id`
        """
        sql = f"SELECT {owner}, key, kind, value FROM {table}"
        parameters = ()
        if owner_id is not None:
            sql += f" WHERE {owner} = ?"
            parameters = (owner_id,)
        properties: dict[int, dict[str, Any]] = {}
        for owner_value, key, kind, value in self._db.execute(sql + " ORDER BY rowid", parameters):
            properties.setdefault(owner_value, {})[key] = self._property_value(table, key, kind, value)
        return properties

    def _property_value(self, table: str, key: str, kind: str, value: Any) -> Any:
        expected = _PROPERTY_KINDS.get(kind)
        if expected is None or type(value) is not expected or (kind == "bool" and value not in (0, 1)):
            raise ValueError(f"{self.path}: {table}: property {key} of kind {kind!r} has value {value!r}")
        return bool(value) if kind == "bool" else value

    def _recording_info(self, row: Sequence, properties: Optional[dict[str, Any]]) -> RecordingInfo:
        name, path, status, error, duration_ms = row
        if path is not None and self._audio_root is not None and not os.path.isabs(path):
            path = os.path.join(self._audio_root, path)
        return RecordingInfo(
            name=name, path=path, status=status, error=error, duration=_from_ms(duration_ms), properties=properties
        )

    def _segment(
        self, table: str, segment_id: int, row: Sequence
    ) -> Union[HypothesisSegmentArrays, ReferenceSegmentArrays]:
        properties_table = _PROPERTY_TABLES[table]
        if properties_table not in self._segment_properties:
            self._segment_properties[properties_table] = self._properties(properties_table, "segment_id")
        properties = self._segment_properties[properties_table].get(segment_id)
        name = row[0]
        try:
            if table == "hypothesis_segments":
                return self._decode_hypothesis(*row, properties=properties)
            return self._decode_reference(*row, properties=properties)
        except _FormatError as e:
            raise ValueError(f"{self.path}: segment {name}: {e}") from None

    def _decode_blobs(self, num_tokens: int, blobs: list) -> list:
        if self._decompress is not None:
            blobs = [None if b is None else self._decompress(b) for b in blobs]
        if num_tokens == 0 and any(b is not None for b in blobs):
            raise _FormatError("per-token values for 0 tokens")
        return blobs

    @staticmethod
    def _check_segment_times(start_ms: Optional[int], end_ms: Optional[int]):
        if start_ms is not None and not 0 <= start_ms <= _UINT32_MAX:
            raise _FormatError(f"start {start_ms} ms out of range")
        if end_ms is not None and not 0 <= end_ms <= _UINT32_MAX:
            raise _FormatError(f"end {end_ms} ms out of range")

    @staticmethod
    def _token_list(tokens: Optional[bytes], num_tokens: int) -> list[str]:
        if num_tokens == 0:
            return []
        if tokens is None:
            raise _FormatError("tokens missing")
        token_list = tokens.decode("utf-8").split("\0")
        if len(token_list) != num_tokens:
            raise _FormatError(f"{len(token_list)} tokens instead of {num_tokens}")
        return token_list

    @staticmethod
    def _times(start_deltas: bytes, durations: bytes, num_tokens: int) -> tuple[np.ndarray, np.ndarray]:
        # uint32 arithmetic wraps around at 2**32 as the spec defines; every result is a valid time
        starts_ms = _values(start_deltas, _U4, num_tokens, "start_deltas").cumsum(dtype=np.uint32)
        ends_ms = starts_ms + _values(durations, _U4, num_tokens, "durations").astype(np.int64)
        return starts_ms, ends_ms

    def _common_fields(self, num_tokens: int, speaker, channels, lang_codes, token_speakers) -> dict[str, Any]:
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
        return {
            "channels": channel_array,
            "lang_codes": lang_list,
            "token_speakers": speaker_list,
            "speaker": segment_speaker,
        }

    def _decode_hypothesis(
        self, name, n_best_index, start_ms, end_ms, speaker, num_tokens, *blobs, properties
    ) -> HypothesisSegmentArrays:
        blobs = self._decode_blobs(num_tokens, list(blobs))
        tokens, start_deltas, durations, channels, lang_codes, token_speakers, confidences, am_scores, lm_scores = blobs
        self._check_segment_times(start_ms, end_ms)
        token_list = self._token_list(tokens, num_tokens)
        if num_tokens == 0:
            starts_ms, ends_ms = np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
        else:
            if start_deltas is None or durations is None:
                raise _FormatError("times missing")
            starts_ms, ends_ms = self._times(start_deltas, durations, num_tokens)
        return HypothesisSegmentArrays(
            name=name,
            n_best_index=n_best_index,
            start=_from_ms(start_ms),
            end=_from_ms(end_ms),
            properties=properties,
            tokens=token_list,
            starts_ms=starts_ms,
            ends_ms=ends_ms,
            **self._common_fields(num_tokens, speaker, channels, lang_codes, token_speakers),
            confidences=None if confidences is None else _decode_confidences(confidences, num_tokens, "confidences"),
            am_scores=None if am_scores is None else _values(am_scores, _F4, num_tokens, "am_scores"),
            lm_scores=None if lm_scores is None else _values(lm_scores, _F4, num_tokens, "lm_scores"),
        )

    def _decode_reference(
        self, name, start_ms, end_ms, speaker, num_tokens, *blobs, properties
    ) -> ReferenceSegmentArrays:
        blobs = self._decode_blobs(num_tokens, list(blobs))
        (tokens, start_deltas, durations, timed, channels, lang_codes, token_speakers, entities, entity_confidences) = (
            blobs
        )
        self._check_segment_times(start_ms, end_ms)
        if (start_ms is None) != (end_ms is None):
            raise _FormatError("only one of start and end")
        token_list = self._token_list(tokens, num_tokens)
        starts_ms = ends_ms = timed_array = None
        if (start_deltas is None) != (durations is None):
            raise _FormatError("only one of start_deltas and durations")
        if start_deltas is not None:
            starts_ms, ends_ms = self._times(start_deltas, durations, num_tokens)
        if timed is not None:
            if start_deltas is None:
                raise _FormatError("timed without times")
            timed_values = _values(timed, _U1, num_tokens, "timed")
            if timed_values.max() > 1:
                raise _FormatError("timed value other than 0 and 1")
            timed_array = timed_values.astype(bool)
        entity_list = None
        if entities is not None:
            try:
                entity_list = [self._entities[i] for i in _values(entities, _U4, num_tokens, "entities").tolist()]
            except KeyError as e:
                raise _FormatError(f"unknown entity index {e}") from None
        entity_confidence_array = None
        if entity_confidences is not None:
            entity_confidence_array = _decode_confidences(entity_confidences, num_tokens, "entity_confidences")
            has_confidence = ~np.isnan(entity_confidence_array)
            if has_confidence.any() and (
                entity_list is None or any(e is None for e, c in zip(entity_list, has_confidence) if c)
            ):
                raise _FormatError("entity confidence without entity")
        return ReferenceSegmentArrays(
            name=name,
            start=_from_ms(start_ms),
            end=_from_ms(end_ms),
            properties=properties,
            tokens=token_list,
            starts_ms=starts_ms,
            ends_ms=ends_ms,
            timed=timed_array,
            **self._common_fields(num_tokens, speaker, channels, lang_codes, token_speakers),
            entities=entity_list,
            entity_confidences=entity_confidence_array,
        )


class _FormatError(Exception):
    """
    A segment violates the format; turned into a ValueError naming the file and segment.
    """


_U1, _U2, _U4, _F4 = np.dtype("u1"), np.dtype("<u2"), np.dtype("<u4"), np.dtype("<f4")


def _values(data: bytes, dtype: np.dtype, num_tokens: int, what: str) -> np.ndarray:
    if len(data) != num_tokens * dtype.itemsize:
        raise _FormatError(f"{what} does not hold {num_tokens} values")
    return np.frombuffer(data, dtype=dtype)


_SEGMENT_TABLES = {TimedTranscript: "hypothesis_segments", ReferenceTranscript: "reference_segments"}
_PROPERTY_TABLES = {
    "hypothesis_segments": "hypothesis_segment_properties",
    "reference_segments": "reference_segment_properties",
}
_SEGMENT_COLUMNS = {
    "hypothesis_segments": (
        "name, n_best_index, start_ms, end_ms, speaker, num_tokens, tokens, start_deltas, durations, "
        "channels, lang_codes, token_speakers, confidences, am_scores, lm_scores"
    ),
    "reference_segments": (
        "name, start_ms, end_ms, speaker, num_tokens, tokens, start_deltas, durations, timed, "
        "channels, lang_codes, token_speakers, entities, entity_confidences"
    ),
}
_PROPERTY_KINDS = {"str": str, "int": int, "float": float, "bool": int}
_RECORDING_FIELDS = ("path", "status", "error", "duration", "properties")
_HYPOTHESIS_PER_TOKEN_FIELDS = (
    "starts_ms",
    "ends_ms",
    "channels",
    "lang_codes",
    "token_speakers",
    "confidences",
    "am_scores",
    "lm_scores",
)
_REFERENCE_PER_TOKEN_FIELDS = (
    "starts_ms",
    "ends_ms",
    "timed",
    "channels",
    "lang_codes",
    "token_speakers",
    "entities",
    "entity_confidences",
)


def _decode_confidences(data: bytes, num_tokens: int, what: str) -> np.ndarray:
    encoded = _values(data, _U2, num_tokens, what)
    missing = encoded == _CONFIDENCE_NONE
    # every code above the scale must be the "no confidence" marker
    if np.count_nonzero(encoded > _CONFIDENCE_SCALE) != np.count_nonzero(missing):
        raise _FormatError(f"{what}: code out of range")
    confidences = encoded / _CONFIDENCE_SCALE
    confidences[missing] = math.nan
    return confidences
