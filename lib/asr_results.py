__all__ = [
    "Seconds",
    "ChannelIndex",
    "Properties",
    "SpeakerInfo",
    "TranscriptBase",
    "TimedTranscript",
    "ReferenceTranscript",
    "Transcript",
    "Segment",
    "Recording",
    "BatchSegment",
    "BatchRecording",
    "stream_asr_results",
    "parse_asr_results",
    "write_asr_results",
]

from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    model_validator,
)

from i6_core.util import uopen


Seconds = NonNegativeFloat  # >= 0.0
ChannelIndex = PositiveInt  # >= 1
# generic key/value metadata, e.g. {"age": "30-40", "snr": 12.5, "native": True}
# int and bool are listed separately so that 30 and true are not read back as 30.0 and 1.0;
# NaN and infinity are rejected as JSON has no representation for them.
Properties = dict[str, str | int | Annotated[float, Field(allow_inf_nan=False)] | bool]


class _StrictModel(BaseModel):
    """
    Base of all models in this module: unknown keys are rejected instead of silently dropped.
    """

    model_config = ConfigDict(extra="forbid")


class SpeakerInfo(_StrictModel):
    """
    Information about a single speaker.

    Within one results file a speaker id identifies one speaker: every occurrence of the same id must come with the
    same name, gender and properties. This is checked when reading and writing results files.
    """

    id: str = Field(default=..., description="Stable identifier for the speaker.")
    name: Optional[str] = Field(default=None, description="Human-readable speaker name/label, if available.")
    gender: Optional[str] = Field(default=None, description="Gender information about the speaker, if available.")
    properties: Optional[Properties] = Field(
        default=None,
        description="Further information about the speaker, e.g. age, dialect or education.",
    )


def _check_start_before_end(start: Optional[float], end: Optional[float]):
    if start is not None and end is not None and start > end:
        raise ValueError("Start time must be smaller or equal to end time.")


class TranscriptBase(_StrictModel):
    """
    Fields shared by all kinds of transcripts: one snippet of text with channel, language and speaker information.
    Abstract: only subclasses such as TimedTranscript and ReferenceTranscript can be instantiated.
    """

    type: str  # placeholder, so `type` is the first key in the JSON

    text: str = Field(
        default=...,
        description="Word(s) of the transcript.",
    )
    channel: ChannelIndex = Field(
        default=1,
        description="Audio channel of the recording that the text belongs to (starting with 1).",
    )

    lang_code: Optional[str] = Field(
        default=None,
        description="Language code for the text if available.",
    )
    speaker: Optional[SpeakerInfo] = Field(
        default=None,
        description="Optional speaker information.",
    )

    def model_post_init(self, context):
        if type(self) is TranscriptBase:
            raise TypeError("TranscriptBase is abstract, use TimedTranscript or ReferenceTranscript")


class TimedTranscript(TranscriptBase):
    """
    One snippet of recognized text with timing, scores and other metadata.
    """

    type: Literal["tt"] = "tt"

    text: str = Field(
        default=...,
        description="Recognized word(s).",
    )

    start: Seconds = Field(
        default=...,
        description="Start time in seconds from the beginning of the recording.",
    )
    end: Seconds = Field(
        default=...,
        description="End time in seconds from the beginning of recording.",
    )

    am_score: Optional[float] = Field(
        default=None,
        description="Optional AM score",
    )
    lm_score: Optional[float] = Field(
        default=None,
        description="Optional LM score",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score between 0 and 1.",
    )

    @model_validator(mode="after")
    def check_times(self):
        _check_start_before_end(self.start, self.end)
        return self


class ReferenceTranscript(TranscriptBase):
    """
    Evaluation-reference text with optional timing and an optional named-entity label with its model score.
    Information about the speaker belongs into `speaker`, about the segment or recording into their `properties`.
    """

    type: Literal["rt"] = "rt"

    text: str = Field(default=..., description="Reference word or phrase.")
    start: Optional[Seconds] = Field(
        default=None,
        description="Aligned start time, if real reference timing is available.",
    )
    end: Optional[Seconds] = Field(
        default=None,
        description="Aligned end time, if real reference timing is available.",
    )
    entity: Optional[str] = Field(
        default=None,
        description="Named-entity label, for example PERSON or DATE_TIME.",
    )
    entity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 of the model that assigned `entity`, if it was computed by one.",
    )

    @model_validator(mode="after")
    def check_times(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Reference start and end time must either both be set or both be omitted.")
        if self.entity_score is not None and self.entity is None:
            raise ValueError("entity_score is given without entity.")
        _check_start_before_end(self.start, self.end)
        return self


Transcript = Annotated[TimedTranscript | ReferenceTranscript, Field(discriminator="type")]


class Segment(_StrictModel):
    """
    A segment of the recording. Segments are used to group multiple transcripts into an n-best list.

    In a reference file, the segments define what is scored:

    - a segment without transcripts means that the correct output is no words,
    - parts of the recording that are not covered by any segment are not scored (as IGNORE_TIME_SEGMENT_IN_SCORING
      in STM files),
    - a segment without start and end covers the whole recording, i.e. from 0 to Recording.duration.

    Reference segments have no n_best_index, and either both a start and an end time or neither.
    """

    type: Literal["seg"] = "seg"

    name: str = Field(default=..., description="Identifier for the segment")
    start: Optional[Seconds] = Field(
        default=None, description="Start time in seconds from the beginning of the recording."
    )
    end: Optional[Seconds] = Field(default=None, description="End time in seconds from the beginning of the recording.")
    n_best_index: Optional[NonNegativeInt] = Field(
        default=None,
        description="Index in the n-best list for this segment. For segments to be grouped into an n-best list the segment name should match.",
    )

    speaker: Optional[SpeakerInfo] = Field(
        default=None,
        description="Optional speaker information.",
    )
    properties: Optional[Properties] = Field(
        default=None,
        description="Further information about the segment, e.g. its domain.",
    )

    @model_validator(mode="after")
    def check_segment_times(self):
        _check_start_before_end(self.start, self.end)
        return self


class Recording(_StrictModel):
    """
    Information about a recording.
    """

    type: Literal["rec"] = "rec"

    name: str = Field(default=..., description="Name of the recording")
    path: Optional[PathLike] = Field(default=None, description="Optional path of the recording file.")
    duration: Optional[Seconds] = Field(
        default=None,
        description="Duration of the recording in seconds, if known. "
        "It is the end of reference segments that have no times.",
    )
    properties: Optional[Properties] = Field(
        default=None,
        description="Further information about the recording, e.g. its domain.",
    )


class BatchSegment(Segment):
    """
    Batched version of a segment, containing the list of transcripts for this segment.

    Transcripts given as dicts must contain the `type` discriminator ("tt" for TimedTranscript, "rt" for
    ReferenceTranscript).
    """

    transcripts: list[Transcript] = Field(
        default=...,
        description="List of transcripts for this segment.",
    )

    @model_validator(mode="after")
    def check_transcripts(self):
        if any(isinstance(t, ReferenceTranscript) for t in self.transcripts):
            _check_reference_segment(self)
        return self


def _check_reference_segment(segment: Segment):
    if segment.n_best_index is not None:
        raise ValueError(f"reference segment {segment.name!r} has an n_best_index, which has no meaning for references")
    if (segment.start is None) != (segment.end is None):
        raise ValueError(f"reference segment {segment.name!r} must have both start and end time, or neither")


class BatchRecording(Recording):
    """
    Batched version of a recording, containing the list of segment alternatives.
    Each list of alternatives represents an n-best list.
    """

    alternatives: list[list[BatchSegment]] = Field(default=..., description="List of segment alternatives.")


class _SpeakerConsistencyChecker:
    """
    Checks that a speaker id always comes with the same speaker information.
    """

    def __init__(self):
        self._speakers: dict[str, SpeakerInfo] = {}

    def check(self, speaker: Optional[SpeakerInfo]):
        if speaker is None:
            return
        known = self._speakers.setdefault(speaker.id, speaker)
        if known != speaker:
            raise ValueError(
                f"speaker {speaker.id!r} given with different name, gender or properties: {known!r} vs. {speaker!r}"
            )


def _check_transcript_type(transcript_type: type[TranscriptBase]):
    if not (isinstance(transcript_type, type) and issubclass(transcript_type, TranscriptBase)) or (
        transcript_type is TranscriptBase
    ):
        raise TypeError(f"transcript_type must be e.g. TimedTranscript or ReferenceTranscript, got {transcript_type!r}")


def stream_asr_results(
    path: str | PathLike,
    transcript_type: type[TranscriptBase] = TimedTranscript,
) -> Generator[Recording | Segment | TranscriptBase]:
    """
    Stream transcripts from a file, by default ASR hypotheses.

    A file contains only one kind of transcript: a transcript line of another type raises a ValueError.
    Segments belong to the most recent Recording line. A recording name that appears again later in the file
    starts a new Recording record; merging such revisits is left to the caller (see parse_asr_results).
    Raises a ValueError if a speaker id is given with different name, gender or properties (see SpeakerInfo).

    :param path: Path to the input file.
    :param transcript_type: Kind of transcript in the file, e.g. TimedTranscript or ReferenceTranscript.
    :yield: Recording, Segment, and `transcript_type` objects.
    """
    _check_transcript_type(transcript_type)
    record_adapter = TypeAdapter(Annotated[Recording | Segment | transcript_type, Field(discriminator="type")])
    in_recording: bool = False
    current_segment: Optional[Segment] = None
    speakers = _SpeakerConsistencyChecker()
    with uopen(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                # skip empty lines
                continue

            # validate JSON directly into the union type
            record = record_adapter.validate_json(line)
            match record:
                case Recording():
                    in_recording = True
                    current_segment = None
                case Segment():
                    assert in_recording, f"Encountered a Segment outside of a recording in line {line_no}"
                    current_segment = record
                    if transcript_type is ReferenceTranscript:
                        _check_reference_segment(record)
                    speakers.check(record.speaker)
                case TranscriptBase():
                    assert in_recording, f"Encountered a transcript outside of a Recording in line {line_no}"
                    assert current_segment is not None, (
                        f"Encountered a transcript outside of a Segment in line {line_no}"
                    )
                    speakers.check(record.speaker)
            yield record


def _n_best_sort_key(seg: Segment) -> tuple[bool, int]:
    """
    Sort key for the segments of an n-best list. Segments without n_best_index come first, in input order.
    """
    return seg.n_best_index is not None, seg.n_best_index or 0


def _parse_transcripts(records: Iterable) -> list[BatchRecording]:
    """
    Groups streamed records into BatchRecording objects.

    A recording name that appears again later in the file is merged into the first BatchRecording of that name:
    its segment groups are appended in file order. Raises a ValueError if the repeated recording has a different path,
    duration or properties.
    """
    result = []
    recordings_by_name: dict[str, BatchRecording] = {}
    current_recording = None
    current_alternatives = None
    for record in records:
        if isinstance(record, Recording):
            if current_alternatives:
                current_alternatives.sort(key=_n_best_sort_key)
            if record.name in recordings_by_name:
                current_recording = recordings_by_name[record.name]
                known = (current_recording.path, current_recording.duration, current_recording.properties)
                given = (record.path, record.duration, record.properties)
                if known != given:
                    raise ValueError(
                        f"recording {record.name!r} given again with different path, duration or properties: "
                        f"{known!r} vs. {given!r}"
                    )
            else:
                current_recording = BatchRecording(**record.model_dump(), alternatives=[])
                result.append(current_recording)
                recordings_by_name[record.name] = current_recording
            current_alternatives = None
        elif isinstance(record, Segment):
            batch_segment = BatchSegment(**record.model_dump(), transcripts=[])
            previous_segment_name = current_alternatives[-1].name if current_alternatives else None
            if batch_segment.name != previous_segment_name:
                if current_alternatives:
                    current_alternatives.sort(key=_n_best_sort_key)
                current_alternatives = []
                current_recording.alternatives.append(current_alternatives)
            current_alternatives.append(batch_segment)
        elif isinstance(record, TranscriptBase):
            current_alternatives[-1].transcripts.append(record)

    if current_alternatives:
        current_alternatives.sort(key=_n_best_sort_key)
    return result


def parse_asr_results(
    path: str | PathLike,
    transcript_type: type[TranscriptBase] = TimedTranscript,
) -> list[BatchRecording]:
    """
    Parse transcripts, by default ASR hypotheses, into BatchRecording objects. Same rules as stream_asr_results.

    A recording name that appears again later in the file is merged into the first BatchRecording of that name:
    its segment groups are appended in file order. Raises a ValueError if the repeated recording has a different path,
    duration or properties.

    :param path: Path to the input file.
    :param transcript_type: Kind of transcript in the file, e.g. TimedTranscript or ReferenceTranscript.
    :return: List of BatchRecording objects containing `transcript_type` transcripts.
    """
    return _parse_transcripts(stream_asr_results(path, transcript_type))


def _to_json_line(record: Recording | Segment | TranscriptBase, exclude: Optional[set[str]] = None) -> str:
    """
    Serialize a record as one JSON line. The `type` discriminator is always written, even if it was only set by
    default, as it is required to parse the line again.
    """
    record = record.model_copy(update={"type": record.type})
    return record.model_dump_json(exclude_unset=True, exclude_none=True, exclude=exclude) + "\n"


def write_asr_results(
    results: list[BatchRecording],
    path: str | PathLike,
    transcript_type: type[TranscriptBase] = TimedTranscript,
):
    """
    Write transcripts, by default ASR hypotheses, to a file.

    Raises a ValueError before writing anything if a speaker id is given with different name, gender or
    properties (see SpeakerInfo), or if a reference segment has an n_best_index, and a TypeError before writing
    anything if a transcript is not of `transcript_type`.

    :param results: List of BatchRecording objects.
    :param path: Path to the output file.
    :param transcript_type: Kind of transcript to write, e.g. TimedTranscript or ReferenceTranscript.
    """
    _check_transcript_type(transcript_type)
    speakers = _SpeakerConsistencyChecker()
    for batch_recording in results:
        for alt in batch_recording.alternatives:
            for batch_seg in alt:
                if transcript_type is ReferenceTranscript:
                    _check_reference_segment(batch_seg)
                speakers.check(batch_seg.speaker)
                for transcript in batch_seg.transcripts:
                    if type(transcript) is not transcript_type:
                        raise TypeError(f"expected {transcript_type.__name__}, got {type(transcript).__name__}")
                    speakers.check(transcript.speaker)

    with uopen(path, "wt", encoding="utf-8") as f:
        for batch_recording in results:
            alternatives = batch_recording.alternatives
            recording = Recording(**batch_recording.model_dump(exclude={"alternatives"}))
            f.write(_to_json_line(recording))

            for alt in alternatives:
                if alt:
                    seg_name = alt[0].name
                    assert all(seg.name == seg_name for seg in alt), (
                        "Segments in an alternative must have the same name."
                    )
                for batch_seg in alt:
                    transcripts = batch_seg.transcripts
                    seg = Segment(**batch_seg.model_dump(exclude={"transcripts"}))
                    f.write(_to_json_line(seg))
                    for transcript in transcripts:
                        f.write(_to_json_line(transcript))
