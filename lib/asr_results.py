__all__ = [
    "Seconds",
    "ChannelIndex",
    "SpeakerInfo",
    "TranscriptBase",
    "TimedTranscript",
    "ReferenceTranscript",
    "Transcript",
    "Segment",
    "Recording",
    "ASRResultsStream",
    "ReferenceTranscriptStream",
    "BatchSegment",
    "BatchRecording",
    "stream_asr_results",
    "stream_reference_transcripts",
    "parse_asr_results",
    "parse_reference_transcripts",
    "write_asr_results",
    "write_reference_transcripts",
]

from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
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


class SpeakerInfo(BaseModel):
    """
    Information about a single speaker.

    Within one results file a speaker id identifies one speaker: every occurrence of the same id must come with the
    same name and gender. This is checked when reading and writing results files.
    """

    id: str = Field(default=..., description="Stable identifier for the speaker.")
    name: Optional[str] = Field(default=None, description="Human-readable speaker name/label, if available.")
    gender: Optional[str] = Field(default=None, description="Gender information about the speaker, if available.")


def _check_start_before_end(start: Optional[float], end: Optional[float]):
    if start is not None and end is not None and start > end:
        raise ValueError("Start time must be smaller or equal to end time.")


class TranscriptBase(BaseModel):
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
        description="Channel index of inside of the recognized recording (starting with 1).",
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
        # runs for every way of creating a model (constructor, model_validate, model_construct, ...)
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
    """Evaluation-reference text with optional timing, an optional named-entity label and other meta data."""

    type: Literal["rtt"] = "rtt"

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
    gender: Optional[str] = Field(default=None, description="Speaker gender label from the reference.")
    age: Optional[str] = Field(default=None, description="Speaker age label from the reference.")
    dialect: Optional[str] = Field(default=None, description="Speaker dialect label from the reference.")
    education: Optional[str] = Field(default=None, description="Speaker education label from the reference.")
    domain: Optional[str] = Field(default=None, description="Domain label from the reference.")

    @model_validator(mode="after")
    def check_times(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Reference start and end time must either both be set or both be omitted.")
        _check_start_before_end(self.start, self.end)
        return self


Transcript = Annotated[TimedTranscript | ReferenceTranscript, Field(discriminator="type")]


class Segment(BaseModel):
    """
    A segment of the recording. Segments are used to group multiple transcripts into an n-best list.
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


class Recording(BaseModel):
    """
    Information about a recording.
    """

    type: Literal["rec"] = "rec"

    name: str = Field(default=..., description="Name of the recording")
    path: Optional[PathLike] = Field(default=None, description="Optional path of the recording file.")


ASRResultsStream = Annotated[Recording | Segment | TimedTranscript, Field(discriminator="type")]
ReferenceTranscriptStream = Annotated[
    Recording | Segment | ReferenceTranscript,
    Field(discriminator="type"),
]


class BatchSegment(Segment):
    """
    Batched version of a segment, containing the list of transcripts for this segment.
    """

    transcripts: list[Transcript] = Field(
        default=...,
        description="List of transcripts for this segment.",
    )


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
        if (known.name, known.gender) != (speaker.name, speaker.gender):
            raise ValueError(f"speaker {speaker.id!r} given with different name or gender: {known!r} vs. {speaker!r}")


def _stream_transcripts(path: str | PathLike, stream_type) -> Generator:
    """
    Reads the file line by line, parsing each line as a JSON object and validating it against `stream_type`.

    Segments belong to the most recent Recording line. A recording name that appears again later in the file
    starts a new Recording record; merging such revisits is left to the caller (see _parse_transcripts).
    Raises a ValueError if a speaker id is given with different name or gender (see SpeakerInfo).
    """
    record_adapter = TypeAdapter(stream_type)
    in_recording: bool = False
    in_segment: bool = False
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
                    in_segment = False
                case Segment():
                    assert in_recording, f"Encountered a Segment outside of a recording in line {line_no}"
                    in_segment = True
                    speakers.check(record.speaker)
                case TranscriptBase():
                    assert in_recording, f"Encountered a transcript outside of a Recording in line {line_no}"
                    assert in_segment, f"Encountered a transcript outside of a Segment in line {line_no}"
                    speakers.check(record.speaker)
            yield record


def stream_asr_results(path: str | PathLike) -> Generator[ASRResultsStream]:
    """
    Stream ASR hypotheses from a file.

    Segments belong to the most recent Recording line. A recording name that appears again later in the file
    starts a new Recording record; merging such revisits is left to the caller (see parse_asr_results).
    Raises a ValueError if a speaker id is given with different name or gender (see SpeakerInfo).

    :param path: Path to the input file.
    :yield: Recording, Segment, and TimedTranscript objects.
    """
    yield from _stream_transcripts(path, ASRResultsStream)


def stream_reference_transcripts(path: str | PathLike) -> Generator[ReferenceTranscriptStream]:
    """
    Stream evaluation-reference transcripts from a file. Same rules as stream_asr_results.

    :param path: Path to the input file.
    :yield: Recording, Segment, and ReferenceTranscript objects.
    """
    yield from _stream_transcripts(path, ReferenceTranscriptStream)


def _n_best_sort_key(seg: Segment) -> tuple[bool, int]:
    """
    Sort key for the segments of an n-best list. Segments without n_best_index come first, in input order.
    """
    return seg.n_best_index is not None, seg.n_best_index or 0


def _parse_transcripts(records: Iterable) -> list[BatchRecording]:
    """
    Groups streamed records into BatchRecording objects.

    A recording name that appears again later in the file is merged into the first BatchRecording of that name:
    its segment groups are appended in file order. Raises a ValueError if the repeated recording has a different path.
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
                if current_recording.path != record.path:
                    raise ValueError(
                        f"recording {record.name!r} given again with different path: "
                        f"{current_recording.path!r} vs. {record.path!r}"
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


def parse_asr_results(path: str | PathLike) -> list[BatchRecording]:
    """
    Parse ASR hypotheses into BatchRecording objects.

    A recording name that appears again later in the file is merged into the first BatchRecording of that name:
    its segment groups are appended in file order. Raises a ValueError if the repeated recording has a different path.

    :param path: Path to the input file.
    :return: List of BatchRecording objects containing ASR hypotheses.
    """
    return _parse_transcripts(stream_asr_results(path))


def parse_reference_transcripts(path: str | PathLike) -> list[BatchRecording]:
    """
    Parse evaluation references into BatchRecording objects. Same rules as parse_asr_results.

    :param path: Path to the input file.
    :return: List of BatchRecording objects containing reference transcripts.
    """
    return _parse_transcripts(stream_reference_transcripts(path))


def _to_json_line(record: Recording | Segment | TranscriptBase) -> str:
    """
    Serialize a record as one JSON line. The `type` discriminator is always written, even if it was only set by
    default, as it is required to parse the line again.
    """
    record = record.model_copy(update={"type": record.type})
    return record.model_dump_json(exclude_unset=True, exclude_none=True) + "\n"


def _write_transcripts(
    results: list[BatchRecording],
    path: str | PathLike,
    transcript_type: type[TranscriptBase],
):
    """
    Raises a ValueError before writing anything if a speaker id is given with different name or gender
    (see SpeakerInfo), and a TypeError before writing anything if a transcript is not of `transcript_type`.
    """
    speakers = _SpeakerConsistencyChecker()
    for batch_recording in results:
        for alt in batch_recording.alternatives:
            for batch_seg in alt:
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


def write_asr_results(results: list[BatchRecording], path: str | PathLike):
    """
    Write ASR hypotheses to a file.

    Raises a ValueError before writing anything if a speaker id is given with different name or gender
    (see SpeakerInfo), and a TypeError if a transcript is not a TimedTranscript.

    :param results: List of BatchRecording objects containing ASR hypotheses.
    :param path: Path to the output file.
    """
    _write_transcripts(results, path, TimedTranscript)


def write_reference_transcripts(results: list[BatchRecording], path: str | PathLike):
    """
    Write evaluation-reference transcripts to a file. Same checks as write_asr_results, for ReferenceTranscript.

    :param results: List of BatchRecording objects containing reference transcripts.
    :param path: Path to the output file.
    """
    _write_transcripts(results, path, ReferenceTranscript)
