import sqlite3
from pathlib import Path

import numpy as np
import pytest

from i6_core.lib.asr_results import (
    BatchRecording,
    BatchSegment,
    ReferenceTranscript,
    SpeakerInfo,
    TimedTranscript,
    parse_asr_results,
    write_asr_results,
)
from i6_core.lib.asr_results_db import (
    APPLICATION_ID,
    ASRResultsDBReader,
    ASRResultsDBWriter,
    ReferenceSegmentArrays,
    HypothesisSegmentArrays,
)

DATA_DIR = Path(__file__).parent


def _dump(recordings: list[BatchRecording]) -> list[dict]:
    """
    Compare models by their JSON content, as `write_asr_results` writes them.
    """
    return [r.model_dump(mode="json", exclude_none=True) for r in recordings]


def _rich_recordings() -> list[BatchRecording]:
    alice = SpeakerInfo(id="spk1", name="Alice", gender="female")
    bob = SpeakerInfo(id="spk2")
    return [
        BatchRecording(
            name="rec-a",
            path="/data/audio/rec-a.wav",
            alternatives=[
                [
                    BatchSegment(
                        name="rec-a/1",
                        start=0.5,
                        end=2.25,
                        n_best_index=0,
                        speaker=alice,
                        transcripts=[
                            TimedTranscript(
                                text="Hello,",
                                start=0.5,
                                end=1.0,
                                channel=2,
                                lang_code="en",
                                speaker=alice,
                                confidence=0.9876,
                                am_score=12.5,
                                lm_score=-3.25,
                            ),
                            TimedTranscript(
                                text="wörld ☃\nline",  # non-ASCII, a space and a newline inside a token
                                start=1.0,
                                end=2.25,
                                channel=2,
                                lang_code="en",
                                speaker=alice,
                                am_score=7.0,
                            ),
                            # punctuation as zero-length token
                            TimedTranscript(text=".", start=2.25, end=2.25, channel=2, lang_code="en", speaker=alice),
                        ],
                    ),
                    BatchSegment(
                        name="rec-a/1",
                        start=0.5,
                        end=2.25,
                        n_best_index=1,
                        transcripts=[TimedTranscript(text="yellow", start=0.5, end=2.25, confidence=0.5)],
                    ),
                ],
                [BatchSegment(name="rec-a/2", start=3.0, end=4.0, n_best_index=0, transcripts=[])],
                [
                    BatchSegment(
                        name="rec-a/3",
                        speaker=bob,
                        transcripts=[TimedTranscript(text="", start=5.0, end=5.5, speaker=bob)],
                    )
                ],
            ],
        ),
        BatchRecording(name="rec-b", alternatives=[]),
    ]


@pytest.mark.parametrize("codec", ["zstd", "none"])
@pytest.mark.parametrize("sample", ["sample.jsonl", "multiple_recordings.jsonl"])
def test_roundtrip_sample_jsonl(tmp_path, codec, sample):
    recordings = parse_asr_results(DATA_DIR / sample)
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec=codec) as writer:
        for recording in recordings:
            writer.add_recording(recording)

    with ASRResultsDBReader(db_path) as reader:
        assert len(reader) == len(recordings)
        assert reader.meta["codec"] == codec
        assert _dump(list(reader.iter_recordings())) == _dump(recordings)


@pytest.mark.parametrize("codec", ["zstd", "none"])
def test_roundtrip_all_fields(tmp_path, codec):
    recordings = _rich_recordings()
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec=codec, compression_level=19, meta={"model": "test-model"}) as writer:
        for recording in recordings:
            writer.add_recording(recording)

    with ASRResultsDBReader(db_path) as reader:
        assert reader.meta["model"] == "test-model"
        assert reader.recording_names() == ["rec-a", "rec-b"]
        assert _dump(list(reader.iter_recordings())) == _dump(recordings)
        assert _dump([reader.get_recording("rec-a")]) == _dump(recordings[:1])
        assert "rec-b" in reader and "rec-c" not in reader
        with pytest.raises(KeyError):
            reader.get_recording("rec-c")


def test_jsonl_export_matches(tmp_path):
    """
    Reading from the DB and writing JSONL gives the same file as writing JSONL directly.
    """
    recordings = _rich_recordings()
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        for recording in recordings:
            writer.add_recording(recording)
    write_asr_results(recordings, tmp_path / "direct.jsonl")
    with ASRResultsDBReader(db_path) as reader:
        from_db = list(reader.iter_recordings())
    write_asr_results(from_db, tmp_path / "via_db.jsonl")
    assert (tmp_path / "via_db.jsonl").read_text() == (tmp_path / "direct.jsonl").read_text()
    assert parse_asr_results(tmp_path / "via_db.jsonl") == from_db


def test_arrays_and_recording_fields(tmp_path):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, audio_root="/data/audio") as writer:
        writer.add_recording_arrays(
            "call-1",
            [
                HypothesisSegmentArrays(
                    name="call-1/1",
                    tokens=["so", "yes", "."],
                    starts_ms=np.array([100, 250, 400]),
                    ends_ms=np.array([250, 400, 400]),
                    confidences=np.array([0.5, np.nan, 1.0]),
                )
            ],
            path="/data/audio/b.1/ca/call-1.wav",
            duration=12.345,
        )
        writer.add_recording_arrays("call-2", [], path="/elsewhere/call-2.wav", status="ASR_ERROR", error="boom")

    with ASRResultsDBReader(db_path) as reader:
        # stored relative to audio_root, returned absolute
        assert reader.execute("SELECT path FROM recordings WHERE name = 'call-1'").fetchone()[0] == "b.1/ca/call-1.wav"
        info = reader.get_recording_arrays("call-1")
        assert info.path == "/data/audio/b.1/ca/call-1.wav"
        assert info.status == "OK" and info.duration == 12.345
        (segment,) = info.segments
        assert segment.tokens == ["so", "yes", "."]
        assert segment.starts_ms.tolist() == [100, 250, 400]
        assert segment.ends_ms.tolist() == [250, 400, 400]
        assert segment.confidences[0] == 0.5 and np.isnan(segment.confidences[1]) and segment.confidences[2] == 1.0
        assert segment.am_scores is None and segment.lm_scores is None

        error = reader.get_recording_arrays("call-2")
        assert (error.path, error.status, error.error, error.segments) == (
            "/elsewhere/call-2.wav",
            "ASR_ERROR",
            "boom",
            [],
        )
        failed = [name for (name,) in reader.execute("SELECT name FROM recordings WHERE status != 'OK'")]
        assert failed == ["call-2"]
        # recordings in file order with their segments
        assert [(i.name, len(i.segments)) for i in reader.iter_recording_arrays()] == [("call-1", 1), ("call-2", 0)]
        assert [i.name for i in reader.iter_recording_arrays(["call-2", "call-1"])] == ["call-2", "call-1"]


def test_file_header(tmp_path):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path):
        pass
    db = sqlite3.connect(db_path)
    assert db.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
    assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    db.execute("PRAGMA user_version = 2")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="format version 2"):
        ASRResultsDBReader(db_path)

    other = tmp_path / "other.db"
    sqlite3.connect(other).execute("CREATE TABLE t (x)").connection.close()
    with pytest.raises(ValueError, match="not an ASR results DB"):
        ASRResultsDBReader(other)


def test_writer_rejects_invalid_input(tmp_path):
    db_path = tmp_path / "results.db"

    def segment(**kwargs):
        values = {"name": "s", "tokens": ["a"], "starts_ms": np.array([0]), "ends_ms": np.array([1])}
        return HypothesisSegmentArrays(**{**values, **kwargs})

    with ASRResultsDBWriter(db_path) as writer:
        with pytest.raises(ValueError, match="NUL"):
            writer.add_recording_arrays("r1", [segment(tokens=["a\0b"])])
        with pytest.raises(ValueError, match="1-based"):
            writer.add_recording_arrays("r2", [segment(channels=np.array([0]))])
        with pytest.raises(ValueError, match="start > end"):
            writer.add_recording_arrays("r3", [segment(starts_ms=np.array([5]), ends_ms=np.array([1]))])
        with pytest.raises(ValueError, match="values for 1 tokens"):
            writer.add_recording_arrays("r4", [segment(confidences=np.array([0.1, 0.2]))])
        with pytest.raises(ValueError, match="between 0 and 1"):
            writer.add_recording_arrays("r5", [segment(confidences=np.array([1.5]))])
        with pytest.raises(ValueError, match="values for 1 tokens"):
            writer.add_recording_arrays("r6", [segment(lang_codes=["en", "de"])])
        writer.add_recording_arrays("ok", [segment()])
        with pytest.raises(ValueError, match="different name, gender or properties"):
            writer.add_recording_arrays("r7", [segment(speaker=SpeakerInfo(id="x", name="A"))])
            writer.add_recording_arrays("r8", [segment(speaker=SpeakerInfo(id="x", name="B"))])
        # a rejected recording that introduced a speaker and a language must not leave them behind
        with pytest.raises(ValueError, match="NUL"):
            writer.add_recording_arrays(
                "r9",
                [
                    segment(lang_codes=["fr"], token_speakers=[SpeakerInfo(id="y")]),
                    segment(tokens=["a\0b"]),
                ],
            )
        writer.add_recording_arrays("r10", [segment(lang_codes=["fr"], token_speakers=[SpeakerInfo(id="y")])])

    with pytest.raises(FileExistsError):
        ASRResultsDBWriter(db_path)
    # rejected recordings are not written, not even partially
    with ASRResultsDBReader(db_path) as reader:
        assert reader.recording_names() == ["ok", "r7", "r10"]
        assert reader.execute("SELECT COUNT(*) FROM hypothesis_segments").fetchone()[0] == 3
        (r10,) = reader.get_recording_arrays("r10").segments
        assert r10.lang_codes == ["fr"] and r10.token_speakers == [SpeakerInfo(id="y")]


def test_per_token_fields(tmp_path):
    """
    channel, lang_code and speaker of the transcripts are stored per token, and only if not all default.
    """
    alice, bob = SpeakerInfo(id="alice"), SpeakerInfo(id="bob", gender="male")
    transcripts = [
        TimedTranscript(text="hi", start=0, end=1, channel=1, lang_code="en", speaker=alice),
        TimedTranscript(text="hallo", start=1, end=2, channel=2, lang_code="de", speaker=bob),
        TimedTranscript(text="ok", start=2, end=3, channel=2),
        TimedTranscript(text="ja", start=3, end=4, lang_code="de", speaker=bob),
    ]
    recordings = [
        BatchRecording(name="mixed", alternatives=[[BatchSegment(name="s", speaker=alice, transcripts=transcripts)]]),
        BatchRecording(
            name="default",
            alternatives=[[BatchSegment(name="s", transcripts=[TimedTranscript(text="x", start=0, end=1)])]],
        ),
    ]
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        for recording in recordings:
            writer.add_recording(recording)
    with ASRResultsDBReader(db_path) as reader:
        assert _dump(list(reader.iter_recordings())) == _dump(recordings)
        (mixed,) = reader.get_recording_arrays("mixed").segments
        assert mixed.channels.tolist() == [1, 2, 2, 1]
        assert mixed.lang_codes == ["en", "de", None, "de"]
        assert mixed.token_speakers == [alice, bob, None, bob]
        assert mixed.speaker == alice
        (default,) = reader.get_recording_arrays("default").segments
        assert (default.channels, default.lang_codes, default.token_speakers) == (None, None, None)
        nulls = reader.execute(
            "SELECT channels IS NULL, lang_codes IS NULL, token_speakers IS NULL FROM hypothesis_segments ORDER BY id"
        ).fetchall()
        assert nulls == [(0, 0, 0), (1, 1, 1)]
        assert sorted(code for (code,) in reader.execute("SELECT lang_code FROM languages")) == ["de", "en"]


def test_failed_write_removes_file(tmp_path):
    db_path = tmp_path / "results.db"
    with pytest.raises(RuntimeError):
        with ASRResultsDBWriter(db_path) as writer:
            writer.add_recording_arrays("r", [])
            raise RuntimeError("job died")
    assert not db_path.exists()


def test_unordered_token_times(tmp_path):
    """
    Start times are stored as uint32 deltas that wrap around for tokens that are not ordered by start.
    """
    db_path = tmp_path / "results.db"
    starts, ends = np.array([5000, 1000, 1000, 2**32 - 2, 0]), np.array([6000, 1500, 1000, 2**32 - 1, 2**32 - 1])
    with ASRResultsDBWriter(db_path) as writer:
        writer.add_recording_arrays(
            "r", [HypothesisSegmentArrays(name="s", tokens=["c", "a", "b", "z", "all"], starts_ms=starts, ends_ms=ends)]
        )
    with ASRResultsDBReader(db_path) as reader:
        (segment,) = reader.get_recording_arrays("r").segments
        assert segment.starts_ms.tolist() == starts.tolist()
        assert segment.ends_ms.tolist() == ends.tolist()


def test_value_limits(tmp_path):
    def segment(**kwargs):
        values = {"name": "s", "tokens": ["a"], "starts_ms": np.array([0]), "ends_ms": np.array([1])}
        return HypothesisSegmentArrays(**{**values, **kwargs})

    with ASRResultsDBWriter(tmp_path / "results.db") as writer:
        for bad in (
            segment(start=float("inf")),
            segment(end=1e306),  # finite, but infinite in ms
            segment(end=4294967.296),  # one ms beyond uint32
            segment(n_best_index=2**63),
            segment(tokens=["\ud800"]),  # lone surrogate
            segment(channels=np.array([65536])),
            segment(am_scores=np.array([1e39])),  # finite, but beyond float32
        ):
            with pytest.raises(ValueError):
                writer.add_recording_arrays("r", [bad])
        writer.add_recording_arrays("r", [segment(end=4294967.295, n_best_index=2**63 - 1)])


def test_scores_have_float32_precision(tmp_path):
    """
    Scores that are float32 values read back exactly, others are rounded to float32.
    """
    from_float32 = float(np.float32(-123.4567))  # what a float32 model score becomes as a Python float
    scores = [from_float32, -123.4567, 0.1]
    assert from_float32 != -123.4567  # -123.4567 and 0.1 are not exact in float32
    recording = BatchRecording(
        name="r",
        alternatives=[
            [
                BatchSegment(
                    name="s",
                    start=0.0,
                    end=3.0,
                    transcripts=[
                        TimedTranscript(text=f"w{i}", start=float(i), end=i + 1.0, am_score=score, lm_score=score)
                        for i, score in enumerate(scores)
                    ],
                )
            ]
        ],
    )
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        writer.add_recording(recording)
    with ASRResultsDBReader(db_path) as reader:
        transcript = reader.get_recording("r").alternatives[0][0].transcripts
    expected = [float(np.float32(score)) for score in scores]
    assert expected[0] == from_float32
    assert [t.am_score for t in transcript] == expected
    assert [t.lm_score for t in transcript] == expected


def test_confidences_round_half_to_even(tmp_path):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec="none") as writer:
        writer.add_recording_arrays(
            "r",
            [
                HypothesisSegmentArrays(
                    name="s",
                    tokens=["a", "b", "c"],
                    starts_ms=np.array([0, 0, 0]),
                    ends_ms=np.array([0, 0, 0]),
                    confidences=np.array([0.00005, 0.00025, 0.00035]),  # x10000: exactly 0.5, 2.5, 3.5
                )
            ],
        )
    (blob,) = sqlite3.connect(db_path).execute("SELECT confidences FROM hypothesis_segments").fetchone()
    assert np.frombuffer(blob, "<u2").tolist() == [0, 2, 4]


@pytest.mark.parametrize(
    "sql, message",
    [
        ("UPDATE hypothesis_segments SET num_tokens = 3", "tokens instead of 3"),
        ("UPDATE hypothesis_segments SET durations = X'01000000'", "durations does not hold 2 values"),
        ("UPDATE hypothesis_segments SET channels = X'00000100'", "channel 0"),
        ("UPDATE hypothesis_segments SET confidences = X'1127FFFF'", "confidences: code out of range"),  # 10001
        ("UPDATE hypothesis_segments SET lang_codes = X'05000000'", "unknown lang_code or speaker index"),
        ("UPDATE hypothesis_segments SET end_ms = -1", "end -1 ms out of range"),
        ("UPDATE hypothesis_segments SET tokens = X'FF0062'", "segment s: tokens are not valid UTF-8"),
    ],
)
def test_reader_rejects_malformed_segments(tmp_path, sql, message):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec="none") as writer:
        writer.add_recording_arrays(
            "r",
            [
                HypothesisSegmentArrays(
                    name="s", tokens=["a", "b"], starts_ms=np.array([0, 10]), ends_ms=np.array([5, 20])
                )
            ],
        )
    db = sqlite3.connect(db_path)
    db.execute(sql)
    db.commit()
    db.close()
    with ASRResultsDBReader(db_path) as reader:
        with pytest.raises(ValueError, match=message):
            reader.get_recording_arrays("r")


def test_reader_rejects_corrupt_zstd_blob(tmp_path):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec="zstd") as writer:
        writer.add_recording_arrays(
            "r",
            [
                HypothesisSegmentArrays(
                    name="s", tokens=["a", "b"], starts_ms=np.array([0, 10]), ends_ms=np.array([5, 20])
                )
            ],
        )
    db = sqlite3.connect(db_path)
    db.execute("UPDATE hypothesis_segments SET tokens = X'78797A78797A78797A'")  # not a zstd frame
    db.commit()
    db.close()
    with ASRResultsDBReader(db_path) as reader:
        with pytest.raises(ValueError, match="segment s: cannot decompress blob"):
            reader.get_recording_arrays("r")


def test_recording_added_in_several_calls(tmp_path):
    """
    Results of several recordings written as they come in: segments of different recordings interleave in the file.
    """

    def segment(name, start):
        return HypothesisSegmentArrays(
            name=name, tokens=[name], starts_ms=np.array([start]), ends_ms=np.array([start + 1])
        )

    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, audio_root="/data/audio") as writer:
        writer.add_recording_arrays("r1", [segment("r1/1", 0)], path="/data/audio/r1.wav")
        writer.add_recording_arrays("r2", [segment("r2/1", 0)])
        writer.add_recording_arrays("r1", [segment("r1/2", 10)])  # continued: no path needed
        writer.add_recording_arrays("r2", [segment("r2/2", 10)], path=None, status="OK")
        with pytest.raises(ValueError, match="different path"):
            writer.add_recording_arrays("r1", [segment("r1/3", 20)], path="/data/audio/other.wav")
        with pytest.raises(ValueError, match="join the n-best alternatives"):
            writer.add_recording_arrays("r1", [segment("r1/2", 20)])
    with ASRResultsDBReader(db_path) as reader:
        assert reader.recording_names() == ["r1", "r2"]
        by_name = {info.name: info for info in reader.iter_recording_arrays()}
        assert [s.name for s in by_name["r1"].segments] == ["r1/1", "r1/2"]
        assert [s.name for s in by_name["r2"].segments] == ["r2/1", "r2/2"]
        assert by_name["r1"].path == "/data/audio/r1.wav"
        assert [s.name for s in reader.get_recording_arrays("r1").segments] == ["r1/1", "r1/2"]


def test_reader_rejects_orphan_segments(tmp_path):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        writer.add_recording_arrays("r", [])
    db = sqlite3.connect(db_path)
    db.execute("INSERT INTO hypothesis_segments (recording_id, name, num_tokens) VALUES (99, 'lost', 0)")
    db.commit()
    db.close()
    with ASRResultsDBReader(db_path) as reader:
        with pytest.raises(ValueError, match="belongs to no recording"):
            list(reader.iter_recording_arrays())


def test_audio_root_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must be absolute"):
        ASRResultsDBWriter(tmp_path / "bad.db", audio_root="data/audio")
    db_path = tmp_path / "results.db"
    paths = {
        "below": "/data/audio/b.1/x.wav",
        "outside": "/elsewhere/y.wav",
        "relative": "rel/z.wav",  # made absolute against the current directory
        "root itself": "/data/audio",
        "sibling prefix": "/data/audio2/w.wav",  # shares the string prefix but is not below the root
    }
    with ASRResultsDBWriter(db_path, audio_root="/data/audio/") as writer:
        for name, path in paths.items():
            writer.add_recording_arrays(name, [], path=path)
    with ASRResultsDBReader(db_path) as reader:
        stored = dict(reader.execute("SELECT name, path FROM recordings"))
        read = {info.name: info.path for info in reader.iter_recording_arrays()}
    assert stored["below"] == "b.1/x.wav"
    expected = {**paths, "relative": str(tmp_path / "rel/z.wav")}
    assert read == expected


def test_writer_rejects_unrepresentable_n_best_structures(tmp_path):
    def seg(name, n_best_index=0):
        return BatchSegment(name=name, n_best_index=n_best_index, transcripts=[])

    with ASRResultsDBWriter(tmp_path / "results.db") as writer:
        for alternatives, message in (
            ([[], [seg("a")]], "empty group"),
            ([[seg("a"), seg("b")]], "mixes segment names"),
            ([[seg("a")], [seg("a", 1)]], "neighbouring n-best groups"),
        ):
            with pytest.raises(ValueError, match=message):
                writer.add_recording(BatchRecording(name="r", alternatives=alternatives))
        # unsorted groups are fine; they are read back sorted, as parse_asr_results does
        writer.add_recording(BatchRecording(name="r", alternatives=[[seg("a", 1), seg("a", 0)], [seg("b")]]))
    with ASRResultsDBReader(tmp_path / "results.db") as reader:
        (recording,) = reader.iter_recordings()
        assert [[(s.name, s.n_best_index) for s in group] for group in recording.alternatives] == [
            [("a", 0), ("a", 1)],
            [("b", 0)],
        ]


def _reference_recordings() -> list[BatchRecording]:
    carol = SpeakerInfo(id="carol", gender="female", properties={"age": "30-40", "native": True})
    return [
        BatchRecording(
            name="ref-a",
            path="/data/audio/ref-a.wav",
            duration=12.5,
            properties={"domain": "finance", "snr": 12.5, "channels": 2, "noisy": False},
            alternatives=[
                [
                    BatchSegment(
                        name="ref-a/1",
                        start=0.5,
                        end=3.0,
                        speaker=carol,
                        properties={"topic": "loans"},
                        transcripts=[
                            ReferenceTranscript(text="I", start=0.5, end=0.75, speaker=carol),
                            ReferenceTranscript(
                                text="New York", start=0.75, end=1.5, entity="LOCATION", entity_confidence=0.93
                            ),
                            ReferenceTranscript(text="Bank", start=1.5, end=2.0, entity="ORGANIZATION"),
                        ],
                    )
                ],
                # partly timed, a channel and a lang_code
                [
                    BatchSegment(
                        name="ref-a/2",
                        start=3.0,
                        end=5.0,
                        transcripts=[
                            ReferenceTranscript(text="yes", start=3.0, end=3.5, channel=2, lang_code="en"),
                            ReferenceTranscript(text="uh", channel=2),
                        ],
                    )
                ],
                # no times at all, and a segment whose correct output is no words
                [BatchSegment(name="ref-a/3", transcripts=[ReferenceTranscript(text="hello")])],
                [BatchSegment(name="ref-a/4", start=6.0, end=7.0, transcripts=[])],
            ],
        ),
        BatchRecording(name="ref-b", alternatives=[]),
    ]


@pytest.mark.parametrize("codec", ["zstd", "none"])
def test_roundtrip_references(tmp_path, codec):
    recordings = _reference_recordings()
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec=codec) as writer:
        for recording in recordings:
            writer.add_recording(recording, ReferenceTranscript)
    with ASRResultsDBReader(db_path) as reader:
        assert reader.transcript_types == [ReferenceTranscript]
        # the kind is found without transcript_type, as only references are in the file
        assert _dump(list(reader.iter_recordings())) == _dump(recordings)
        assert _dump([reader.get_recording("ref-a")]) == _dump(recordings[:1])
        segments = reader.get_recording_arrays("ref-a").segments
        assert [s.timed is None and s.starts_ms is not None for s in segments] == [True, False, False, False]
        assert segments[1].timed.tolist() == [True, False]
        assert segments[2].starts_ms is None and segments[2].timed is None
        assert segments[0].entities == [None, "LOCATION", "ORGANIZATION"]
        assert segments[0].entity_confidences[1] == 0.93 and np.isnan(segments[0].entity_confidences[2])
        nulls = reader.execute(
            "SELECT start_deltas IS NULL, timed IS NULL, entities IS NULL FROM reference_segments ORDER BY id"
        ).fetchall()
        assert nulls == [(0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, 1)]
        assert reader.execute("SELECT COUNT(*) FROM hypothesis_segments").fetchone()[0] == 0

    # the JSONL format of references gives the same file directly and via the DB
    write_asr_results(recordings, tmp_path / "direct.jsonl", ReferenceTranscript)
    with ASRResultsDBReader(db_path) as reader:
        from_db = list(reader.iter_recordings())
    write_asr_results(from_db, tmp_path / "via_db.jsonl", ReferenceTranscript)
    assert (tmp_path / "via_db.jsonl").read_text() == (tmp_path / "direct.jsonl").read_text()
    assert parse_asr_results(tmp_path / "via_db.jsonl", ReferenceTranscript) == from_db


def test_properties_and_duration(tmp_path):
    """
    Properties of speakers, recordings and segments keep the type of their values; durations are stored in ms.
    """
    speaker = SpeakerInfo(id="s", properties={"count": 30, "ratio": 30.0, "flag": True, "label": "30"})
    recording = BatchRecording(
        name="r",
        duration=1.2345,  # stored as 1235 ms (rounded half to even)
        properties={"big": 2**63 - 1, "neg": -1.5, "off": False},
        alternatives=[
            [
                BatchSegment(
                    name="s1",
                    speaker=speaker,
                    properties={"x": 1},
                    transcripts=[TimedTranscript(text="a", start=0, end=1, speaker=speaker)],
                )
            ]
        ],
    )
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        writer.add_recording(recording)
        for bad in ({"nan": float("nan")}, {"inf": float("inf")}, {"big": 2**63}, {"list": [1]}):
            with pytest.raises(ValueError):
                writer.add_recording_arrays("bad", [], properties=bad)
        with pytest.raises(ValueError, match="different name, gender or properties"):
            other = SpeakerInfo(id="s", properties={"count": 31})
            writer.add_recording_arrays(
                "bad", [HypothesisSegmentArrays(name="x", tokens=[], starts_ms=[], ends_ms=[], speaker=other)]
            )
        with pytest.raises(ValueError, match="different properties"):
            writer.add_recording_arrays("r", [], properties={"big": 1})
    with ASRResultsDBReader(db_path) as reader:
        (read,) = reader.iter_recordings()
        assert read.duration == 1.234
        assert read.properties == recording.properties
        assert [type(v) for v in read.properties.values()] == [int, float, bool]
        segment = read.alternatives[0][0]
        assert segment.properties == {"x": 1}
        assert segment.speaker == speaker and segment.transcripts[0].speaker == speaker
        assert [type(v) for v in segment.speaker.properties.values()] == [int, float, bool, str]
        assert reader.recording_names() == ["r"]


def test_hypotheses_and_references_in_one_file(tmp_path):
    """
    A file may hold recordings of both kinds, but a recording holds only one.
    """
    hypothesis = BatchRecording(
        name="h", alternatives=[[BatchSegment(name="h/1", transcripts=[TimedTranscript(text="a", start=0, end=1)])]]
    )
    reference = BatchRecording(name="r", alternatives=[[BatchSegment(name="r/1", transcripts=[])]])
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path) as writer:
        writer.add_recording(hypothesis)
        writer.add_recording(reference, ReferenceTranscript)
        writer.add_recording(BatchRecording(name="empty", alternatives=[]))
        with pytest.raises(TypeError, match="expected ReferenceTranscript, got TimedTranscript"):
            writer.add_recording(BatchRecording(name="x", alternatives=hypothesis.alternatives), ReferenceTranscript)
        with pytest.raises(ValueError, match="both hypothesis and reference"):
            writer.add_recording_arrays("h", [ReferenceSegmentArrays(name="h/2", tokens=[])])
        with pytest.raises(ValueError, match="both hypothesis and reference"):
            writer.add_recording_arrays(
                "y",
                [
                    ReferenceSegmentArrays(name="y/1", tokens=[]),
                    HypothesisSegmentArrays(name="y/2", tokens=[], starts_ms=[], ends_ms=[]),
                ],
            )
        # the other kind's segments may follow once a recording has none yet
        writer.add_recording_arrays("empty", [ReferenceSegmentArrays(name="empty/1", tokens=["z"])])
    with ASRResultsDBReader(db_path) as reader:
        assert reader.transcript_types == [TimedTranscript, ReferenceTranscript]
        with pytest.raises(ValueError, match="give transcript_type"):
            list(reader.iter_recordings())
        assert [r.name for r in reader.iter_recordings(transcript_type=TimedTranscript)] == ["h"]
        assert [r.name for r in reader.iter_recordings(transcript_type=ReferenceTranscript)] == ["r", "empty"]
        # a single recording is read as its own kind
        assert isinstance(reader.get_recording("h").alternatives[0][0].transcripts[0], TimedTranscript)
        assert reader.get_recording("r").alternatives[0][0].transcripts == []
        with pytest.raises(ValueError, match="no TimedTranscript segments"):
            reader.get_recording("r", TimedTranscript)


def test_reference_rules(tmp_path):
    with ASRResultsDBWriter(tmp_path / "results.db") as writer:
        with pytest.raises(ValueError, match="has an n_best_index"):
            writer.add_recording(
                BatchRecording(name="r", alternatives=[[BatchSegment(name="s", n_best_index=0, transcripts=[])]]),
                ReferenceTranscript,
            )
        for bad, message in (
            (ReferenceSegmentArrays(name="s", tokens=[], start=1.0), "both start and end"),
            (ReferenceSegmentArrays(name="s", tokens=[], start=2.0, end=1.0), "start > end"),
            (ReferenceSegmentArrays(name="s", tokens=["a"], starts_ms=np.array([0])), "both be given"),
            (ReferenceSegmentArrays(name="s", tokens=["a"], timed=np.array([True])), "timed is given without"),
            (
                ReferenceSegmentArrays(name="s", tokens=["a"], entity_confidences=np.array([0.5])),
                "entity_confidence without entity",
            ),
            (
                ReferenceSegmentArrays(name="s", tokens=["a"], entities=["X"], entity_confidences=np.array([1.5])),
                "between 0 and 1",
            ),
        ):
            with pytest.raises(ValueError, match=message):
                writer.add_recording_arrays("r", [bad])
        with pytest.raises(ValueError, match="start > end"):
            writer.add_recording_arrays(
                "h", [HypothesisSegmentArrays(name="s", tokens=[], starts_ms=[], ends_ms=[], start=2, end=1)]
            )


@pytest.mark.parametrize(
    "sql, message",
    [
        ("UPDATE reference_segments SET timed = X'0002'", "timed value other than 0 and 1"),
        ("UPDATE reference_segments SET entities = X'0000000009000000'", "unknown entity index"),
        ("UPDATE reference_segments SET entity_confidences = X'1027FFFF', entities = NULL", "without entity"),
        ("UPDATE reference_segments SET durations = NULL", "only one of start_deltas and durations"),
        ("UPDATE recording_properties SET kind = 'float'", "of kind 'float' has value 1"),
    ],
)
def test_reader_rejects_malformed_references(tmp_path, sql, message):
    db_path = tmp_path / "results.db"
    with ASRResultsDBWriter(db_path, codec="none") as writer:
        writer.add_recording_arrays(
            "r",
            [
                ReferenceSegmentArrays(
                    name="s",
                    tokens=["a", "b"],
                    starts_ms=np.array([0, 0]),
                    ends_ms=np.array([5, 0]),
                    timed=np.array([True, False]),
                    entities=[None, "X"],
                )
            ],
            properties={"n": 1},
        )
    db = sqlite3.connect(db_path)
    db.execute(sql)
    db.commit()
    db.close()
    with ASRResultsDBReader(db_path) as reader:
        with pytest.raises(ValueError, match=message):
            reader.get_recording_arrays("r")
