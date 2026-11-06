import json
from pathlib import Path

import pytest

from i6_core.lib.asr_results import (
    BatchRecording,
    BatchSegment,
    SpeakerInfo,
    Recording,
    Segment,
    TimedTranscript,
    parse_asr_results,
    stream_asr_results,
    write_asr_results,
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
    with pytest.raises(ValueError, match="speaker 'a' given with different name or gender"):
        list(stream_asr_results(input_file))
    with pytest.raises(ValueError, match="speaker 'a' given with different name or gender"):
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
    with pytest.raises(ValueError, match="speaker 'a' given with different name or gender"):
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
