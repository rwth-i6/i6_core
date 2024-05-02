import os
import tempfile
from typing import Any
import pytest
from sisyphus import setup_path
from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
import i6_core.lib.corpus as libcorpus


Path = setup_path(__package__)


@pytest.mark.skip(reason="Helper class only used in file.")
class CorpusCreatorHelper:
    @staticmethod
    def _create_corpus_with_structure(corpus_dict: dict[str, Any]) -> libcorpus.Corpus:
        """
        Creates a simple corpus from its definition in dictionary form.
        The dictionary can have the following keys and with the respective type:
        - "name": str
        - "recordings": list[dict[str, Any]]: can have the following keys:
            - "name": str
            - "segments": list[str]
        - "subcorpora": list[dict[str, Any]]: recursive definition of a corpus.

        :param corpus_dict: Definition of a corpus in dictionary form.
        :return: Corpus object defined by the corpus dictionary provided.
        """
        corpus = libcorpus.Corpus()
        corpus.name = corpus_dict["name"]
        for recording_dict in corpus_dict.get("recordings", []):
            recording = libcorpus.Recording()
            recording.name = recording_dict["name"]
            for segment_name in recording_dict.get("segments", []):
                segment = libcorpus.Segment()
                segment.name = segment_name
                segment.orth = ""
                recording.add_segment(segment)
            corpus.add_recording(recording)
        for subcorpus_dict in corpus_dict.get("subcorpora", []):
            corpus.add_subcorpus(CorpusCreatorHelper._create_corpus_with_structure(subcorpus_dict))

        return corpus

    @staticmethod
    def _check_corpus_with_structure(corpus: libcorpus.Corpus, corpus_dict: dict[str, Any]) -> bool:
        """
        Asserts that the corpus passed as parameter has the same structure as the simple corpus dictionary provided.

        :param corpus: Corpus for which to check the structure.
        :param corpus_dict: Expected corpus structure.
        :return: True if the corpus matches the given structure.
            Doesn't return False whenever the corpus structure doesn't match, but fails with an assertion.
        """
        # Check that the corpus name matches.
        assert (
            corpus.name == corpus_dict["name"]
        ), f"Corpus name ({corpus.name}) doesn't coincide with provided ({corpus_dict['name']})."

        # Check that the recordings (and their internal segments) match.
        recordings_dicts = corpus_dict.get("recordings", [])
        assert len(corpus.recordings) == len(recordings_dicts), (
            f"Number of recordings in corpus ({len(corpus.recordings)}) not coinciding "
            f"with provided ({len(recordings_dicts)})."
        )
        for recording_dict in recordings_dicts:
            segments_from_dict = recording_dict.get("segments", [])

            # Get the recording with the same name.
            recordings_in_corpus_with_same_name = [
                r for r in corpus.top_level_recordings() if r.name == recording_dict["name"]
            ]
            found_recording = None
            for r in recordings_in_corpus_with_same_name:
                if sorted([segment.name for segment in r.segments]) == sorted(segments_from_dict):
                    found_recording = r
                    break
            assert (
                found_recording
            ), f"No recording in {corpus.fullname()} found with the features provided by the user: {recording_dict}. "

        # Recursively check that the subcorpora match.
        subcorpus_dicts = corpus_dict.get("subcorpora", [])
        for subcorpus_dict in subcorpus_dicts:
            matching_subcorpus = [sc for sc in corpus.top_level_subcorpora() if sc.name == subcorpus_dict["name"]]
            assert (
                len(matching_subcorpus) == 1
            ), f"There's more than one subcorpus matching for the subcorpus name {subcorpus_dict['name']}"
            CorpusCreatorHelper._check_corpus_with_structure(matching_subcorpus[0], subcorpus_dict)

        return True


def test_merge_corpora_job_merge_recursive():
    """
    Merges two hardcoded corpora with the `MergeStrategy.MERGE_RECURSIVE` whose structures are the following:

    CORPUS 1            CORPUS 2
    rec1                rec1
        seg10               seg13
    sc1                 sc1
        rec1                rec1
            seg1..2             seg3..5
        rec2
            seg3..4
    ------------------------------------
    sc2                 sc2
        rec2
            seg5
        rec3
            seg6
                            sc1
                                rec1
                                    seg1
    ------------------------------------
                        sc3
                            rec1
                                seg1
                            rec2

    The expected output structure is the following:
    rec1
        seg10
        seg13
    sc1
        rec1
            seg1..5
        rec2
            seg3..4
    ------------------------------------
    sc2
        rec2
            seg5
        rec3
            seg6
        sc1
            rec1
                seg1
    ------------------------------------
    sc3
        rec1
            seg1
        rec2

    Some notes about the expected output:
    `MergeStrategy.MERGE_RECURSIVE` only extends the segment list of the recordings.
    - The nested subcorpus sc2/sc1 shouldn't be merged with the main subcorpus sc1.
    Moreover, its recording sc2/sc1/rec1 shouldn't be merged with neither rec1 nor sc1/rec1.
    `MergeStrategy.MERGE_RECURSIVE` only takes into consideration the top level subcorpora.
    - It's intended that sc3/rec2 doesn't have any segments.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # First corpus.
        c1_path = os.path.join(tmpdir, "c1.xml.gz")
        c1_structure = {
            "name": "c1",
            "recordings": [
                {"name": "rec1", "segments": ["seg10"]},
            ],
            "subcorpora": [
                {
                    "name": "sc1",
                    "recordings": [
                        {"name": "rec1", "segments": ["seg1", "seg2"]},
                        {"name": "rec2", "segments": ["seg3", "seg4"]},
                    ],
                },
                {
                    "name": "sc2",
                    "recordings": [
                        {"name": "rec2", "segments": ["seg5"]},
                        {"name": "rec3", "segments": ["seg6"]},
                    ],
                },
            ],
        }
        c1 = CorpusCreatorHelper._create_corpus_with_structure(c1_structure)
        c1.dump(c1_path)

        # Second corpus.
        c2_path = os.path.join(tmpdir, "c2.xml.gz")
        c2_structure = {
            "name": "c2",
            "recordings": [
                {"name": "rec1", "segments": ["seg13"]},
            ],
            "subcorpora": [
                {"name": "sc1", "recordings": [{"name": "rec1", "segments": ["seg3", "seg4", "seg5"]}]},
                {
                    "name": "sc2",
                    "subcorpora": [
                        {
                            "name": "sc1",
                            "recordings": [{"name": "rec1", "segments": ["seg1"]}],
                        }
                    ],
                },
                {
                    "name": "sc3",
                    "recordings": [
                        {"name": "rec1", "segments": ["seg1"]},
                        {"name": "rec2"},
                    ],
                },
            ],
        }
        c2 = CorpusCreatorHelper._create_corpus_with_structure(c2_structure)
        c2.dump(c2_path)

        # Merge the first and second corpora.
        merged_corpus_path = os.path.join(tmpdir, "merged_corpus.xml.gz")
        merge_corpora_job = MergeCorporaJob(
            bliss_corpora=[Path(c1_path), Path(c2_path)],
            name="custom_name_for_merged_corpus",
            merge_strategy=MergeStrategy.MERGE_RECURSIVE,
        )
        merge_corpora_job.out_merged_corpus = Path(merged_corpus_path)
        merge_corpora_job.run()

        merged_corpus = libcorpus.Corpus()
        merged_corpus.load(merged_corpus_path)
        expected_merged_corpus_structure = {
            "name": "custom_name_for_merged_corpus",
            "recordings": [
                {"name": "rec1", "segments": ["seg10", "seg13"]},
            ],
            "subcorpora": [
                {
                    "name": "sc1",
                    "recordings": [
                        {"name": "rec1", "segments": ["seg1", "seg2", "seg3", "seg4", "seg5"]},
                        {"name": "rec2", "segments": ["seg3", "seg4"]},
                    ],
                },
                {
                    "name": "sc2",
                    "recordings": [
                        {"name": "rec2", "segments": ["seg5"]},
                        {"name": "rec3", "segments": ["seg6"]},
                    ],
                    "subcorpora": [
                        {
                            "name": "sc1",
                            "recordings": [{"name": "rec1", "segments": ["seg1"]}],
                        }
                    ],
                },
                {
                    "name": "sc3",
                    "recordings": [
                        {"name": "rec1", "segments": ["seg1"]},
                        {"name": "rec2"},
                    ],
                },
            ],
        }
        CorpusCreatorHelper._check_corpus_with_structure(merged_corpus, expected_merged_corpus_structure)
