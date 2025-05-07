__all__ = [
    "FilterSegmentsByListJob",
    "FilterSegmentsByRegexJob",
    "FilterSegmentsByAlignmentConfidenceJob",
    "FilterCorpusBySegmentsJob",
    "FilterCorpusRemoveUnknownWordSegmentsJob",
    "FilterCorpusBySegmentDurationJob",
]

import gzip
import logging
import numpy as np
import re
import xml.etree.cElementTree as ET
from typing import Dict, List, Optional, Union

from i6_core import rasr
from i6_core.lib import corpus
from i6_core.util import chunks, uopen, MultiOutputPath

from sisyphus import *

Path = setup_path(__package__)


def _delete_empty_recordings(corpus: corpus.Corpus, removed_recordings_file: str):
    """
    Deletes all recordings that are empty after the filtering done by some of the jobs in this file.

    :param c: Corpus for which to delete the empty recordings.
    :param removed_recordings_file: File in which to dump all recordings that have been deleted.
    """
    to_delete = []
    for rec in corpus.all_recordings():
        if not rec.segments:
            to_delete.append(rec)

    corpus.remove_recordings(to_delete)
    with open(removed_recordings_file, "w") as f:
        f.write("\n".join(rec.fullname() for rec in to_delete))


class FilterSegmentsByListJob(Job):
    def __init__(self, segment_files: Dict[int, Path], filter_list: Union[List[str], Path], invert_match: bool = False):
        """
        Filters segment list file using a given list of segments, which is either used as black or as white list
        :param segment_files: original segment list files to be filtered
        :param filter_list: list used for filtering or a path to a text file with the entries of that list one per line
        :param invert_match: black list (if False) or white list (if True) usage
        """
        assert isinstance(filter_list, tk.Path) or isinstance(filter_list, list)
        self.segment_files = segment_files
        self.filter_list = filter_list
        self.invert_match = invert_match

        num_segment_lists = len(self.segment_files)
        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i)) for i in range(1, num_segment_lists + 1)
        )
        self.out_segment_path = MultiOutputPath(self, "segments.$(TASK)", self.out_single_segment_files)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        if isinstance(self.filter_list, tk.Path):
            filter_list = [line.rstrip() for line in open(self.filter_list.get_path(), "r")]
        elif isinstance(self.filter_list, list):
            filter_list = self.filter_list
        else:
            assert False

        for idx, segment_file in self.segment_files.items():
            segment_list = [line.rstrip() for line in open(segment_file.get_path(), "r")]
            non_empty = False
            with open(self.out_single_segment_files[idx].get_path(), "wt") as segment_file_filtered:
                for segment in segment_list:
                    if (self.invert_match and segment in filter_list) or (
                        not self.invert_match and segment not in filter_list
                    ):
                        segment_file_filtered.write(segment + "\n")
                        non_empty = True
            if not non_empty:
                logging.warning(
                    "Segment file empty after filtering: {}".format(self.out_single_segment_files[idx].get_path())
                )


class FilterSegmentsByRegexJob(Job):
    def __init__(self, segment_files: Dict[int, Path], filter_regex: str, invert_match: bool = False):
        """
        Filters segment list file using a given regular expression
        :param segment_files: original segment list files to be filtered
        :param filter_regex: regex used for filtering
        :param invert_match: keep segment if regex does not match (if False) or does match (if True)
        """
        self.segment_files = segment_files
        self.filter_regex = filter_regex
        self.invert_match = invert_match

        num_segment_lists = len(self.segment_files)
        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i)) for i in range(1, num_segment_lists + 1)
        )
        self.out_segment_path = MultiOutputPath(self, "segments.$(TASK)", self.out_single_segment_files)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        pattern = re.compile(self.filter_regex)
        for idx, segment_file in self.segment_files.items():
            segment_list = [line.rstrip() for line in open(segment_file.get_path(), "r")]
            output_empty = True
            with open(self.out_single_segment_files[idx].get_path(), "wt") as segment_file_filtered:
                for segment in segment_list:
                    if (self.invert_match and pattern.match(segment)) or (
                        not self.invert_match and not pattern.match(segment)
                    ):
                        segment_file_filtered.write(segment + "\n")
                        output_empty = False
            if output_empty:
                logging.warning(
                    "Segment file empty after filtering: {}".format(self.out_single_segment_files[idx].get_path())
                )


class FilterSegmentsByAlignmentConfidenceJob(Job):
    def __init__(
        self,
        alignment_logs: Dict[int, Path],
        percentile: float,
        crp: Optional[rasr.CommonRasrParameters] = None,
        plot: bool = True,
        absolute_threshold: Optional[float] = None,
    ):
        """
        :param alignment_logs: alignment_job.out_log_file; task_id -> log_file
        :param percentile: percent of alignment segments to keep. should be in (0,100]. for :func:`np.percentile`
        :param crp: used to set the number of output segments. if none, number of alignment log files is used instead.
        :param plot: plot the distribution of alignment scores
        :param absolute_threshold: alignments with score above this number are discarded
        """
        self.alignment_logs = alignment_logs  # alignment_job.log_file
        self.percentile = percentile
        self.absolute_threshold = absolute_threshold
        self.num_segments = len(alignment_logs) if crp is None else crp.concurrent
        self.plot = plot

        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i)) for i in range(1, self.num_segments + 1)
        )
        self.out_segment_path = MultiOutputPath(self, "segments.$(TASK)", self.out_single_segment_files)
        self.out_single_file = self.output_path("filtered.segments")
        if plot:
            self.out_plot_avg = self.output_path("score.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        segment_dict = {}
        for task_id, log_file in self.alignment_logs.items():
            logging.info("Reading: {}".format(log_file))
            file_path = tk.uncached_path(log_file)
            document = ET.parse(uopen(file_path))
            _seg_list = document.findall(".//segment")
            for seg in _seg_list:
                avg = seg.find(".//score/avg")
                segment_dict[seg.attrib["full-name"]] = float(avg.text)
            del document

        logging.info("Scores has {} entries.".format(len(segment_dict)))
        score_np = np.asarray(list(segment_dict.values()))
        logging.info("Max {}; Min {}; Median {}".format(score_np.max(), score_np.min(), np.median(score_np)))
        avg_score_threshold = np.percentile(score_np, self.percentile)
        if np.isnan(avg_score_threshold):
            avg_score_threshold = np.inf
        logging.info("Avg Threshold is {} with percentile {}".format(avg_score_threshold, self.percentile))
        if self.absolute_threshold is not None:
            avg_score_threshold = min(avg_score_threshold, self.absolute_threshold)
        logging.info("Threshold is {}".format(avg_score_threshold))

        if self.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_percentile = np.percentile(score_np, 90)  # there can be huge outliers
            np.clip(score_np, 0, 200, out=score_np)
            plt.hist(score_np, bins=100, range=(0, 200))
            plt.xlabel("Average Maximum-Likelihood Score")
            plt.ylabel("Number of Segments")
            plt.title("Histogram of Alignment Scores")
            plt.savefig(fname=self.out_plot_avg.get_path())

        # Only keep segments that are below the threshold
        filtered_segments = [seg for seg, avg in segment_dict.items() if avg <= avg_score_threshold]
        logging.info("Have {} entries after filtering.".format(len(filtered_segments)))

        for idx, segments in enumerate(chunks(filtered_segments, self.num_segments)):
            with open(self.out_single_segment_files[idx + 1].get_path(), "wt") as segment_file:
                for segment in segments:
                    segment_file.write(segment + "\n")

        with open(self.out_single_file.get_path(), "wt") as segment_file:
            for segment in filtered_segments:
                segment_file.write(segment + "\n")


class FilterCorpusBySegmentsJob(Job):
    __sis_hash_exclude__ = {"delete_empty_recordings": False}

    def __init__(
        self,
        bliss_corpus: Path,
        segment_file: Union[List[Path], Path],
        compressed: bool = False,
        invert_match: bool = False,
        delete_empty_recordings: bool = False,
    ):
        """
        :param bliss_corpus:
        :param segment_file: a single segment file or a list of segment files
        :param compressed:
        :param invert_match:
        :param delete_empty_recordings: if true, empty recordings will be removed
        """
        self.bliss_corpus = bliss_corpus
        self.segment_file_list = [segment_file] if isinstance(segment_file, tk.Path) else segment_file
        self.invert_match = invert_match
        self.delete_empty_recordings = delete_empty_recordings

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if compressed else ""))
        if self.delete_empty_recordings:
            self.out_removed_recordings = self.output_path("removed_recordings.log")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        segments = []
        for seg in self.segment_file_list:
            with uopen(tk.uncached_path(seg), "rt") as f:
                lines = f.readlines()
                segments += [l.strip() for l in lines]

        logging.info("There are #{} segments in the segment list.".format(len(segments)))
        segments = set(segments)
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))

        for rec in c.all_recordings():
            if self.invert_match:
                rec.segments = [x for x in rec.segments if x.fullname() not in segments and x.name not in segments]
            else:
                rec.segments = [x for x in rec.segments if x.fullname() in segments or x.name in segments]

        if self.delete_empty_recordings:
            # Remove the recordings without segments due to the filtering.
            _delete_empty_recordings(c, self.out_removed_recordings.get_path())

        c.dump(tk.uncached_path(self.out_corpus))


class FilterCorpusRemoveUnknownWordSegmentsJob(Job):
    """
    Filter segments of a bliss corpus if there are unknowns with respect to a given lexicon
    """

    __sis_hash_exclude__ = {
        "all_unknown": None,
        "delete_empty_recordings": False,
        "segment_oov_tolerance": None,
        "recording_oov_tolerance": 1.0,
    }

    def __init__(
        self,
        bliss_corpus: tk.Path,
        bliss_lexicon: tk.Path,
        case_sensitive: bool = False,
        all_unknown: Optional[bool] = None,
        delete_empty_recordings: bool = False,
        segment_oov_tolerance: Optional[float] = None,
        recording_oov_tolerance: float = 1.0,
    ):
        """
        :param bliss_corpus:
        :param bliss_lexicon:
        :param case_sensitive: consider casing for check against lexicon
        :param all_unknown: all words have to be unknown in order for the segment to be discarded
        :param delete_empty_recordings: if true, empty recordings will be removed.
        :param segment_oov_tolerance: maximal word OOV rate for a segment to be kept.
            A value of 0.0 means no single OOV word is allowed, 1.0 means everything is allowed.
        :param maximal percentage of high word OOV rate segments for a recording to be kept.
            A value of 0.0 means a single high segment with an OOV rate above the segment_oov_tolerance will cause the recording to be deleted, 1.0 means no recording will be deleted.
        """
        self.corpus = bliss_corpus
        self.lexicon = bliss_lexicon
        self.case_sensitive = case_sensitive
        assert (
            all_unknown is None or segment_oov_tolerance is None
        ), "`all_unknown` and `segment_oov_tolerance` can't be set in the same time."
        self.all_unknown = all_unknown if all_unknown is not None else True
        self.delete_empty_recordings = delete_empty_recordings
        self.segment_oov_tolerance = segment_oov_tolerance
        self.recording_oov_tolerance = recording_oov_tolerance

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)
        if self.delete_empty_recordings:
            self.out_removed_recordings = self.output_path("removed_recordings.log")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        def maybe_to_lower(s):
            return s if self.case_sensitive else s.lower()

        lex_path = self.lexicon.get_path()
        open_func = gzip.open if lex_path.endswith(".gz") else open
        with open_func(lex_path, "rt") as f:
            lex_root = ET.parse(f)
        vocabulary = set([maybe_to_lower(o.text.strip() if o.text else "") for o in lex_root.findall(".//orth")])
        vocabulary -= {
            maybe_to_lower(o.text.strip() if o.text else "")
            for l in lex_root.findall(".//lemma")
            if l.attrib.get("special") == "unknown"
            for o in l.findall(".//orth")
        }

        c = corpus.Corpus()
        c.load(self.corpus.get_path())
        num_segments_per_recording = {r.fullname(): len(r.segments) for r in c.all_recordings()}

        def unknown_filter(corpus: corpus.Corpus, recording: corpus.Recording, segment: corpus.Segment) -> bool:
            """
            :param corpus: needed to match the filter signature
            :param recording: needed to match the filter signature
            :param segment: segment to filter
            :return: whether the orth of segment contains at least one known word (all_unknown = True) or
                     whether all orths are in the lexicon (all_unknown = False)
            """
            orth = segment.orth
            if not orth:
                return True
            words = [maybe_to_lower(o) for o in orth.strip().split(" ")]
            num_oov_words = sum(1 if w not in vocabulary else 0 for w in words)

            if self.segment_oov_tolerance is None:
                if self.all_unknown:
                    return num_oov_words < len(words)
                else:
                    return num_oov_words == 0
            else:
                return num_oov_words <= len(words) * self.segment_oov_tolerance

        c.filter_segments(unknown_filter)

        if self.recording_oov_tolerance < 1.0:
            recordings_to_be_removed = []
            for r in c.all_recordings():
                num_seg = num_segments_per_recording[r.fullname()]
                new_num_seg = len(r.segments)
                if num_seg and (num_seg - new_num_seg) / num_seg > self.recording_oov_tolerance:
                    recordings_to_be_removed.append(r)

            c.remove_recordings(recordings_to_be_removed)

        if self.delete_empty_recordings:
            # Remove the recordings without segments due to the filtering.
            _delete_empty_recordings(c, self.out_removed_recordings.get_path())

        c.dump(self.out_corpus.get_path())

    @classmethod
    def hash(cls, kwargs):
        kwargs_copy = dict(**kwargs)

        if "all_unknown" in kwargs_copy and kwargs_copy["all_unknown"] is True:
            del kwargs_copy["all_unknown"]

        return super().hash(kwargs_copy)


class FilterCorpusBySegmentDurationJob(Job):
    """
    Removes all segments from all corpus recordings that don't fall within the specified duration boundaries.
    """

    __sis_hash_exclude__ = {"delete_empty_recordings": False}

    def __init__(
        self,
        bliss_corpus: Path,
        min_duration: float = 0.1,
        max_duration: float = 120.0,
        delete_empty_recordings: bool = False,
    ):
        """
        :param bliss_corpus: path of the corpus file
        :param min_duration: minimum duration for a segment to keep (in seconds)
        :param max_duration: maximum duration for a segment to keep (in seconds)
        :param delete_empty_recordings: if true, empty recordings will be removed.
        """
        self.bliss_corpus = bliss_corpus
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.delete_empty_recordings = delete_empty_recordings

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)
        if self.delete_empty_recordings:
            self.out_removed_recordings = self.output_path("removed_recordings.log")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        inf = float("inf")

        def good_duration(corpus, recording, segment):
            l = segment.end - segment.start
            if l == inf:
                return True
            else:
                return l >= self.min_duration and l <= self.max_duration

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        c.filter_segments(good_duration)

        if self.delete_empty_recordings:
            # Remove the recordings without segments due to the filtering.
            _delete_empty_recordings(c, self.out_removed_recordings.get_path())

        c.dump(self.out_corpus.get_path())
