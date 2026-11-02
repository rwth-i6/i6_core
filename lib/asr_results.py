__all__ = [
    "Seconds",
    "ChannelIndex",
    "SpeakerInfo",
    "TimedTranscript",
    "ReferenceTimedTranscript",
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
    """

    id: str = Field(default=..., description="Stable identifier for the speaker.")
    name: Optional[str] = Field(default=None, description="Human-readable speaker name/label, if available.")
    gender: Optional[str] = Field(default=None, description="Gender information about the speaker, if available.")


class TimedTranscript(BaseModel):
    """
    One snippet of text with timing and other metadata.
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
        if self.start > self.end:
            raise ValueError("Start time must be smaller or equal to end time.")
        return self


class ReferenceTimedTranscript(TimedTranscript):
    """Evaluation-reference text with an optional named-entity label and other meta data."""

    type: Literal["rtt"] = "rtt"
    text: str = Field(default=..., description="Reference word or phrase.")
    entity: Optional[str] = Field(
        default=None,
        description="Named-entity label, for example PERSON or DATE_TIME.",
    )
    gender: Optional[str] = Field(default=None, description="Speaker gender label from the reference.")
    age: Optional[str] = Field(default=None, description="Speaker age label from the reference.")
    dialect: Optional[str] = Field(default=None, description="Speaker dialect label from the reference.")
    education: Optional[str] = Field(default=None, description="Speaker education label from the reference.")
    domain: Optional[str] = Field(default=None, description="Domain label from the reference.")


Transcript = Annotated[TimedTranscript | ReferenceTimedTranscript, Field(discriminator="type")]


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
    Recording | Segment | ReferenceTimedTranscript,
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


def _stream_transcripts(path: str | PathLike, stream_type) -> Generator:
    record_adapter = TypeAdapter(stream_type)
    in_recording: bool = False
    in_segment: bool = False
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
                case TimedTranscript():
                    assert in_recording, f"Encountered a TimedTranscript outside of a Recording in line {line_no}"
                    assert in_segment, f"Encountered a TimedTranscript outside of a Segment in line {line_no}"
            yield record


def stream_asr_results(path: str | PathLike) -> Generator[ASRResultsStream]:
    """
    Stream ASR hypotheses from a file.

    :param path: Path to the input file.
    :yield: Recording, Segment, and TimedTranscript objects.
    """
    yield from _stream_transcripts(path, ASRResultsStream)


def stream_reference_transcripts(path: str | PathLike) -> Generator[ReferenceTranscriptStream]:
    """
    Stream evaluation-reference transcripts from a file.

    :param path: Path to the input file.
    :yield: Recording, Segment, and ReferenceTimedTranscript objects.
    """
    yield from _stream_transcripts(path, ReferenceTranscriptStream)


def _parse_transcripts(records: Iterable) -> list[BatchRecording]:
    result = []
    current_recording = None
    current_alternatives = None
    for record in records:
        if isinstance(record, Recording):
            if current_alternatives:
                current_alternatives.sort(key=lambda seg: seg.n_best_index)
            result.append(BatchRecording(**record.model_dump(), alternatives=[]))
            current_recording = result[-1]
            current_alternatives = None
        elif isinstance(record, Segment):
            batch_segment = BatchSegment(**record.model_dump(), transcripts=[])
            previous_segment_name = current_alternatives[-1].name if current_alternatives else None
            if batch_segment.name != previous_segment_name:
                if current_alternatives:
                    current_alternatives.sort(key=lambda seg: seg.n_best_index)
                current_alternatives = []
                current_recording.alternatives.append(current_alternatives)
            current_alternatives.append(batch_segment)
        elif isinstance(record, TimedTranscript):
            current_alternatives[-1].transcripts.append(record)

    if current_alternatives:
        current_alternatives.sort(key=lambda seg: seg.n_best_index)
    return result


def parse_asr_results(path: str | PathLike) -> list[BatchRecording]:
    """
    Parse ASR hypotheses into BatchRecording objects.

    :param path: Path to the input file.
    :return: List of BatchRecording objects containing ASR hypotheses.
    """
    return _parse_transcripts(stream_asr_results(path))


def parse_reference_transcripts(path: str | PathLike) -> list[BatchRecording]:
    """
    Parse evaluation references into BatchRecording objects.

    :param path: Path to the input file.
    :return: List of BatchRecording objects containing reference transcripts.
    """
    return _parse_transcripts(stream_reference_transcripts(path))


def _write_transcripts(
    results: list[BatchRecording],
    path: str | PathLike,
    transcript_type: type[TimedTranscript],
):
    with uopen(path, "wt", encoding="utf-8") as f:
        for batch_recording in results:
            alternatives = batch_recording.alternatives
            recording = Recording(**batch_recording.model_dump(exclude={"alternatives"}))
            f.write(recording.model_dump_json(exclude_unset=True, exclude_none=True) + "\n")

            for alt in alternatives:
                if alt:
                    seg_name = alt[0].name
                    assert all(seg.name == seg_name for seg in alt), (
                        "Segments in an alternative must have the same name."
                    )
                for batch_seg in alt:
                    transcripts = batch_seg.transcripts
                    seg = Segment(**batch_seg.model_dump(exclude={"transcripts"}))
                    f.write(seg.model_dump_json(exclude_unset=True, exclude_none=True) + "\n")
                    for transcript in transcripts:
                        if type(transcript) is not transcript_type:
                            raise TypeError(
                                f"expected {transcript_type.__name__}, got {type(transcript).__name__}"
                            )
                        f.write(transcript.model_dump_json(exclude_unset=True, exclude_none=True) + "\n")


def write_asr_results(results: list[BatchRecording], path: str | PathLike):
    """
    Write ASR hypotheses to a file.

    :param results: List of BatchRecording objects containing ASR hypotheses.
    :param path: Path to the output file.
    """
    _write_transcripts(results, path, TimedTranscript)


def write_reference_transcripts(results: list[BatchRecording], path: str | PathLike):
    """
    Write evaluation-reference transcripts to a file.

    :param results: List of BatchRecording objects containing reference transcripts.
    :param path: Path to the output file.
    """
    _write_transcripts(results, path, ReferenceTimedTranscript)
