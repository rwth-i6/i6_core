import json
from pathlib import Path

import pytest

from i6_core.lib.asr_results import (
    BatchRecording,
    BatchSegment,
    SpeakerInfo,
    Recording,
    ReferenceTranscript,
    Segment,
    TimedTranscript,
    TranscriptBase,
    parse_asr_results,
    parse_reference_transcripts,
    stream_asr_results,
    write_asr_results,
    write_reference_transcripts,
)


@pytest.fixture
def sample_asr_file():
    return Path(__file__).parent / "sample.jsonl"


def test_stream_asr_results(sample_asr_file):
    results = list(stream_asr_results(sample_asr_file))
    assert len(results) == 8
    expected_types = [
        Recording,
        Segment,
        TimedTranscript,
        TimedTranscript,
        Segment,
        TimedTranscript,
        Segment,
        TimedTranscript,
    ]
    assert all(isinstance(record, expected_type) for record, expected_type in zip(results, expected_types))
    assert results[0].name == "rec1"
    assert results[1].name == "seg1"
    assert results[2].text == "hello"
    assert results[3].text == "world"
    assert results[4].name == "seg2"
    assert results[5].text == "bar"
    assert results[6].name == "seg2"
    assert results[7].text == "foo"


def test_parse_asr_results(sample_asr_file):
    recordings = parse_asr_results(sample_asr_file)
    assert len(recordings) == 1
    rec = recordings[0]
    assert isinstance(rec, BatchRecording)
    assert rec.name == "rec1"

    # Check alternatives
    # seg1 has 1 alternative (index 0)
    # seg2 has 2 alternatives (index 1 and 0)

    assert len(rec.alternatives) == 2

    # First segment group (seg1)
    alt1 = rec.alternatives[0]
    assert len(alt1) == 1
    assert alt1[0].name == "seg1"
    assert len(alt1[0].transcripts) == 2
    assert alt1[0].transcripts[0].text == "hello"
    assert alt1[0].transcripts[1].text == "world"

    # Second segment group (seg2)
    # Should be sorted by n_best_index
    alt2 = rec.alternatives[1]
    assert len(alt2) == 2
    assert alt2[0].n_best_index == 0
    assert alt2[0].transcripts[0].text == "bar"  # The one with index 0 has "bar" (based on input order, wait)

    # Input order:
    # seg2 index 1: "foo"
    # seg2 index 0: "bar"
    # Sorted: index 0 ("bar"), index 1 ("foo")

    assert alt2[1].n_best_index == 1
    assert alt2[1].transcripts[0].text == "foo"


@pytest.fixture
def multiple_recordings_file():
    return Path(__file__).parent / "multiple_recordings.jsonl"


def test_parse_asr_results_multiple_recordings(multiple_recordings_file):
    recordings = parse_asr_results(multiple_recordings_file)
    assert len(recordings) == 2
    assert recordings[0].name == "rec1"
    assert recordings[1].name == "rec2"
    assert len(recordings[0].alternatives) == 1
    assert len(recordings[1].alternatives) == 1


def test_write_asr_results(sample_asr_file, tmp_path):
    recordings = parse_asr_results(sample_asr_file)
    output_file = tmp_path / "output.jsonl"
    write_asr_results(recordings, output_file)

    with open(sample_asr_file) as f1, open(output_file) as f2:
        records_f1 = [json.loads(line) for line in f1]
        records_f2 = [json.loads(line) for line in f2]
        assert records_f1 == records_f2


def test_reference_transcript_round_trip(tmp_path):
    input_file = tmp_path / "reference.jsonl"
    input_file.write_text(
        '{"type":"rec","name":"rec1","properties":{"domain":"call-center"}}\n'
        '{"type":"seg","name":"seg1","speaker":{"id":"spk1","gender":"female","properties":{"age":"30-40"}}}\n'
        '{"type":"rtt","text":"in"}\n'
        '{"type":"rtt","text":"New York","entity":"LOCATION"}\n'
    )

    with pytest.raises(ValueError):
        list(stream_asr_results(input_file))

    recordings = parse_reference_transcripts(input_file)
    ordinary_transcript = recordings[0].alternatives[0][0].transcripts[0]
    entity_transcript = recordings[0].alternatives[0][0].transcripts[1]
    assert isinstance(ordinary_transcript, ReferenceTranscript)
    assert ordinary_transcript.entity is None
    assert ordinary_transcript.start is None
    assert ordinary_transcript.end is None
    assert recordings[0].properties == {"domain": "call-center"}
    segment_speaker = recordings[0].alternatives[0][0].speaker
    assert segment_speaker.gender == "female"
    assert segment_speaker.properties == {"age": "30-40"}
    assert isinstance(entity_transcript, ReferenceTranscript)
    assert entity_transcript.text == "New York"
    assert entity_transcript.entity == "LOCATION"

    output_file = tmp_path / "output.jsonl"
    write_reference_transcripts(recordings, output_file)
    assert output_file.read_text() == input_file.read_text()


def test_write_parse_roundtrip_python_models(tmp_path):
    recordings = [
        BatchRecording(
            name="rec1",
            alternatives=[
                [
                    BatchSegment(
                        name="seg1",
                        n_best_index=0,
                        transcripts=[TimedTranscript(text="hello", start=0.0, end=1.0)],
                    ),
                    BatchSegment(
                        name="seg1",
                        n_best_index=1,
                        transcripts=[TimedTranscript(text="yellow", start=0.0, end=1.0, confidence=0.5)],
                    ),
                ]
            ],
        )
    ]
    output_file = tmp_path / "output.jsonl"
    write_asr_results(recordings, output_file)
    assert parse_asr_results(output_file) == recordings


def _write_lines(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_parse_asr_results_without_n_best_index(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1"},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "foo", "start": 0.0, "end": 1.0},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "bar", "start": 0.0, "end": 1.0},
        ],
    )
    (rec,) = parse_asr_results(input_file)
    (alt,) = rec.alternatives
    assert [seg.n_best_index for seg in alt] == [None, None]
    assert [seg.transcripts[0].text for seg in alt] == ["foo", "bar"]


def test_parse_asr_results_mixed_n_best_index(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1"},
            {"type": "seg", "name": "seg1", "n_best_index": 0},
            {"type": "tt", "text": "foo", "start": 0.0, "end": 1.0},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "bar", "start": 0.0, "end": 1.0},
        ],
    )
    (rec,) = parse_asr_results(input_file)
    (alt,) = rec.alternatives
    assert [seg.n_best_index for seg in alt] == [None, 0]
    assert [seg.transcripts[0].text for seg in alt] == ["bar", "foo"]


def test_parse_asr_results_inconsistent_speaker(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1"},
            {"type": "seg", "name": "seg1", "speaker": {"id": "a", "gender": "f"}},
            {"type": "tt", "text": "foo", "start": 0.0, "end": 1.0, "speaker": {"id": "a", "gender": "f"}},
            {"type": "rec", "name": "rec2"},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "bar", "start": 0.0, "end": 1.0, "speaker": {"id": "a"}},
        ],
    )
    with pytest.raises(ValueError, match="speaker 'a' given with different name, gender or properties"):
        list(stream_asr_results(input_file))
    with pytest.raises(ValueError, match="speaker 'a' given with different name, gender or properties"):
        parse_asr_results(input_file)


def test_write_asr_results_inconsistent_speaker(tmp_path):
    recordings = [
        BatchRecording(
            name="rec1",
            alternatives=[
                [
                    BatchSegment(
                        name="seg1",
                        speaker=SpeakerInfo(id="a", name="Alice"),
                        transcripts=[
                            TimedTranscript(text="foo", start=0.0, end=1.0, speaker=SpeakerInfo(id="a", name="Bob"))
                        ],
                    )
                ]
            ],
        )
    ]
    output_file = tmp_path / "output.jsonl"
    with pytest.raises(ValueError, match="speaker 'a' given with different name, gender or properties"):
        write_asr_results(recordings, output_file)
    assert not output_file.exists()


def test_parse_asr_results_consistent_speaker(tmp_path):
    input_file = tmp_path / "input.jsonl"
    speaker = {"id": "a", "name": "Alice", "gender": "f"}
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1"},
            {"type": "seg", "name": "seg1", "speaker": speaker},
            {"type": "tt", "text": "foo", "start": 0.0, "end": 1.0, "speaker": speaker},
            {"type": "rec", "name": "rec2"},
            {"type": "seg", "name": "seg1", "speaker": speaker},
            {"type": "tt", "text": "bar", "start": 0.0, "end": 1.0, "speaker": speaker},
        ],
    )
    assert len(parse_asr_results(input_file)) == 2


def test_parse_asr_results_recording_revisit(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1", "path": "/a.wav"},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "foo", "start": 0.0, "end": 1.0},
            {"type": "rec", "name": "rec2"},
            {"type": "seg", "name": "seg1"},
            {"type": "tt", "text": "bar", "start": 0.0, "end": 1.0},
            {"type": "rec", "name": "rec1", "path": "/a.wav"},
            {"type": "seg", "name": "seg2"},
            {"type": "tt", "text": "baz", "start": 1.0, "end": 2.0},
        ],
    )
    assert [record.name for record in stream_asr_results(input_file) if isinstance(record, Recording)] == [
        "rec1",
        "rec2",
        "rec1",
    ]
    rec1, rec2 = parse_asr_results(input_file)
    assert (rec1.name, rec2.name) == ("rec1", "rec2")
    assert [alt[0].name for alt in rec1.alternatives] == ["seg1", "seg2"]
    assert [alt[0].transcripts[0].text for alt in rec1.alternatives] == ["foo", "baz"]
    assert [alt[0].transcripts[0].text for alt in rec2.alternatives] == ["bar"]


def test_parse_asr_results_recording_revisit_different_path(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1", "path": "/a.wav"},
            {"type": "rec", "name": "rec1", "path": "/b.wav"},
        ],
    )
    with pytest.raises(ValueError, match="recording 'rec1' given again with different path"):
        parse_asr_results(input_file)


def test_write_parse_reference_roundtrip_python_models(tmp_path):
    recordings = [
        BatchRecording(
            name="rec1",
            alternatives=[
                [
                    BatchSegment(
                        name="seg1",
                        transcripts=[
                            ReferenceTranscript(text="in"),
                            ReferenceTranscript(text="New York", entity="LOCATION", start=0.5, end=1.0),
                        ],
                    )
                ]
            ],
        )
    ]
    output_file = tmp_path / "output.jsonl"
    write_reference_transcripts(recordings, output_file)

    parsed = parse_reference_transcripts(output_file)
    assert parsed == recordings


def test_write_rejects_other_transcript_type(tmp_path):
    recordings = [
        BatchRecording(
            name="rec1",
            alternatives=[[BatchSegment(name="seg1", transcripts=[ReferenceTranscript(text="in")])]],
        )
    ]
    output_file = tmp_path / "output.jsonl"
    with pytest.raises(TypeError, match="expected TimedTranscript, got ReferenceTranscript"):
        write_asr_results(recordings, output_file)
    assert not output_file.exists()


def test_reference_transcript_is_not_a_timed_transcript():
    reference = ReferenceTranscript(text="in")
    assert isinstance(reference, TranscriptBase)
    assert not isinstance(reference, TimedTranscript)
    assert not hasattr(reference, "am_score")
    assert not hasattr(reference, "confidence")
    with pytest.raises(ValueError):
        TimedTranscript(text="in")  # times are required for hypotheses


@pytest.mark.parametrize("times", [{"start": 1.0}, {"end": 1.0}])
def test_reference_transcript_requires_both_or_no_times(times):
    with pytest.raises(ValueError, match="both be set or both be omitted"):
        ReferenceTranscript(text="in", **times)


def test_reference_transcript_start_after_end():
    with pytest.raises(ValueError, match="Start time must be smaller or equal to end time"):
        ReferenceTranscript(text="in", start=2.0, end=1.0)


def test_transcript_base_is_abstract():
    with pytest.raises(TypeError, match="TranscriptBase is abstract"):
        TranscriptBase(type="tt", text="in")
    with pytest.raises(TypeError, match="TranscriptBase is abstract"):
        TranscriptBase.model_validate({"type": "tt", "text": "in"})


def test_reference_transcript_has_no_per_word_speaker_fields():
    assert {"gender", "age", "dialect", "education", "domain"}.isdisjoint(ReferenceTranscript.model_fields)


def test_parse_asr_results_inconsistent_speaker_properties(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1"},
            {"type": "seg", "name": "seg1", "speaker": {"id": "a", "properties": {"age": "30-40"}}},
            {
                "type": "tt",
                "text": "foo",
                "start": 0.0,
                "end": 1.0,
                "speaker": {"id": "a", "properties": {"age": "50-60"}},
            },
        ],
    )
    with pytest.raises(ValueError, match="speaker 'a' given with different name, gender or properties"):
        parse_asr_results(input_file)


def test_parse_asr_results_recording_revisit_different_properties(tmp_path):
    input_file = tmp_path / "input.jsonl"
    _write_lines(
        input_file,
        [
            {"type": "rec", "name": "rec1", "properties": {"domain": "news"}},
            {"type": "rec", "name": "rec1", "properties": {"domain": "sports"}},
        ],
    )
    with pytest.raises(ValueError, match="recording 'rec1' given again with different path or properties"):
        parse_asr_results(input_file)


def test_write_parse_roundtrip_properties(tmp_path):
    speaker = SpeakerInfo(id="a", gender="f", properties={"dialect": "bavarian"})
    recordings = [
        BatchRecording(
            name="rec1",
            properties={"domain": "news"},
            alternatives=[
                [
                    BatchSegment(
                        name="seg1",
                        speaker=speaker,
                        properties={"topic": "weather"},
                        transcripts=[TimedTranscript(text="hello", start=0.0, end=1.0, speaker=speaker)],
                    )
                ]
            ],
        )
    ]
    output_file = tmp_path / "output.jsonl"
    write_asr_results(recordings, output_file)
    assert parse_asr_results(output_file) == recordings
