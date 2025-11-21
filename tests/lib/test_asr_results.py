import json
from pathlib import Path

import pytest

from i6_core.lib.asr_results import (
    BatchRecording,
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
