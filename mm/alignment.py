__all__ = ["AlignmentJob", "DumpAlignmentJob", "AMScoresFromAlignmentLogJob", "ComputeTSEJob"]

import logging
import math
import os
import shutil
import statistics
import xml.etree.ElementTree as ET
from typing import Callable, Counter, List, Optional, Tuple, Union

from sisyphus import *

Path = setup_path(__package__)

import i6_core.lib.rasr_cache as rasr_cache
import i6_core.rasr as rasr
import i6_core.util as util

from .flow import alignment_flow, dump_alignment_flow


class AlignmentJob(rasr.RasrCommand, Job):
    """
    Align a dataset with the given feature scorer.
    """

    __sis_hash_exclude__ = {"plot_alignment_scores": False}

    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options=None,
        word_boundaries=False,
        use_gpu=False,
        rtf=1.0,
        extra_config=None,
        extra_post_config=None,
        plot_alignment_scores=False,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param bool use_gpu:
        :param float rtf:
        :param extra_config:
        :param extra_post_config:
        :param plot_alignment_scores: Whether to plot the alignment scores (normalized over time) or not.
            The recommended value is `True`. The default value is `False` for retrocompatibility purposes.
        """
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = AlignmentJob.create_config(**kwargs)
        self.alignment_flow = AlignmentJob.create_flow(**kwargs)
        self.concurrent = crp.concurrent
        self.exe = self.select_exe(crp.acoustic_model_trainer_exe, "acoustic-model-trainer")
        self.feature_scorer = feature_scorer
        self.use_gpu = use_gpu
        self.word_boundaries = word_boundaries
        self.plot_alignment_scores = plot_alignment_scores

        self.out_log_file = self.log_file_output_path("alignment", crp, True)
        self.out_single_alignment_caches = dict(
            (i, self.output_path("alignment.cache.%d" % i, cached=True)) for i in range(1, self.concurrent + 1)
        )
        self.out_alignment_path = util.MultiOutputPath(
            self,
            "alignment.cache.$(TASK)",
            self.out_single_alignment_caches,
            cached=True,
        )
        self.out_alignment_bundle = self.output_path("alignment.cache.bundle", cached=True)
        if self.word_boundaries:
            self.out_single_word_boundary_caches = dict(
                (i, self.output_path("word_boundary.cache.%d" % i, cached=True)) for i in range(1, self.concurrent + 1)
            )
            self.out_word_boundary_path = util.MultiOutputPath(
                self,
                "word_boundary.cache.$(TASK)",
                self.out_single_word_boundary_caches,
                cached=True,
            )
            self.out_word_boundary_bundle = self.output_path("word_boundary.cache.bundle", cached=True)
        if self.plot_alignment_scores:
            self.out_plot_avg = self.output_path("score.png")

        self.rqmt = {
            "time": max(rtf * crp.corpus_duration / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 1 if self.use_gpu else 0,
            "mem": 2,
        }

    def tasks(self):
        rqmt = self.rqmt.copy()
        if isinstance(self.feature_scorer, rasr.GMMFeatureScorer):
            mixture_size = os.stat(tk.uncached_path(self.feature_scorer.config["file"])).st_size / (1024.0**2)
            rqmt["mem"] += int(math.ceil((mixture_size - 200.0) / 750.0))

        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=rqmt, args=range(1, self.concurrent + 1))
        if self.plot_alignment_scores:
            yield Task("plot", resume="plot", rqmt=rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "alignment.config")
        self.alignment_flow.write_to_file("alignment.flow")
        util.write_paths_to_file(self.out_alignment_bundle, self.out_single_alignment_caches.values())
        if self.word_boundaries:
            util.write_paths_to_file(
                self.out_word_boundary_bundle,
                self.out_single_word_boundary_caches.values(),
            )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(self.exe, "alignment.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "alignment.cache.%d" % task_id,
            self.out_single_alignment_caches[task_id].get_path(),
        )
        if self.word_boundaries:
            shutil.move(
                "word_boundary.cache.%d" % task_id,
                self.out_single_word_boundary_caches[task_id].get_path(),
            )

    def plot(self):
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        # Parse the files and search for the average alignment score values (normalized over time).
        alignment_scores = []
        for log_file in self.out_log_file.values():
            logging.info("Reading: {}".format(log_file))
            file_path = log_file.get_path()
            document = ET.parse(util.uopen(file_path))
            _seg_list = document.findall(".//segment")
            for seg in _seg_list:
                avg = seg.find(".//score/avg")
                alignment_scores.append(float(avg.text))
            del document

        np_alignment_scores = np.asarray(alignment_scores)
        higher_percentile = np.percentile(np_alignment_scores, 90)  # There can be huge outliers.
        logging.info(
            f"Max {np_alignment_scores.max()}; min {np_alignment_scores.min()}; median {np.median(np_alignment_scores)}"
        )
        logging.info(f"Total number of segments: {np_alignment_scores.size}; 90-th percentile: {higher_percentile}")

        # Plot the data.
        matplotlib.use("Agg")
        np.clip(np_alignment_scores, np_alignment_scores.min(), higher_percentile, out=np_alignment_scores)
        plt.hist(np_alignment_scores, bins=100)
        plt.xlabel("Average Maximum-Likelihood Score")
        plt.ylabel("Number of Segments")
        plt.title("Histogram of Alignment Scores")
        plt.savefig(fname=self.out_plot_avg.get_path())

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("alignment.log.%d" % task_id)
        util.delete_if_exists("alignment.cache.%d" % task_id)
        if self.word_boundaries:
            util.delete_if_zero("word_boundary.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options,
        word_boundaries,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param extra_config:
        :param extra_post_config:
        :return: config, post_config
        :rtype: (rasr.RasrConfig, rasr.RasrConfig)
        """
        alignment_flow = cls.create_flow(feature_flow)

        # TODO: think about mode
        alignopt = {
            "increase-pruning-until-no-score-difference": True,
            "min-acoustic-pruning": 500,
            "max-acoustic-pruning": 4000,
            "acoustic-pruning-increment-factor": 2,
        }
        if alignment_options is not None:
            alignopt.update(alignment_options)

        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }

        # acoustic model + lexicon for the flow nodes
        for node in alignment_flow.get_node_names_by_filter("speech-alignment"):
            mapping["lexicon"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.lexicon"
                % node
            )
            mapping["acoustic_model"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.acoustic-model"
                % node
            )

        config, post_config = rasr.build_config_from_mapping(crp, mapping, parallelize=True)

        # alignment options for the flow nodes
        for node in alignment_flow.get_node_names_by_filter("speech-alignment"):
            node_config = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[node]

            node_config.aligner = rasr.RasrConfig()
            for k, v in alignopt.items():
                node_config.aligner[k] = v
            feature_scorer.apply_config("model-combination.acoustic-model.mixture-set", node_config, node_config)

            if word_boundaries:
                node_config.store_lattices = True
                node_config.lattice_archive.path = "word_boundary.cache.$(TASK)"

        alignment_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = "alignment.flow"
        post_config["*"].allow_overwrite = True

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, **kwargs):
        return alignment_flow(feature_flow, "alignment.cache.$(TASK)")

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        alignment_flow = cls.create_flow(**kwargs)
        return super().hash(
            {
                "config": config,
                "alignment_flow": alignment_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class DumpAlignmentJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        original_alignment,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Dump Alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = DumpAlignmentJob.create_config(**kwargs)
        self.dump_flow = DumpAlignmentJob.create_flow(**kwargs)
        self.exe = self.select_exe(crp.acoustic_model_trainer_exe, "acoustic-model-trainer")
        self.concurrent = crp.concurrent

        self.out_log_file = self.log_file_output_path("dump", crp, True)
        self.out_single_alignment_caches = dict(
            (i, self.output_path("alignment.cache.%d" % i, cached=True)) for i in range(1, self.concurrent + 1)
        )
        self.out_alignment_path = util.MultiOutputPath(
            self,
            "alignment.cache.$(TASK)",
            self.out_single_alignment_caches,
            cached=True,
        )
        self.out_alignment_bundle = self.output_path("alignment.cache.bundle", cached=True)

        self.rqmt = {
            "time": max(crp.corpus_duration / (50.0 * crp.concurrent), 0.5),
            "cpu": 1,
            "mem": 1,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

    def create_files(self):
        self.write_config(self.config, self.post_config, "dump.config")
        self.dump_flow.write_to_file("dump.flow")
        util.write_paths_to_file(self.out_alignment_bundle, self.out_single_alignment_caches.values())
        self.write_run_script(self.exe, "dump.config")

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "alignment.cache.%d" % task_id,
            self.out_single_alignment_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("dump.log.%d" % task_id)
        util.delete_if_zero("alignment.cache.%d" % task_id)

    @classmethod
    def create_config(cls, crp, extra_config, extra_post_config, **kwargs):
        dump_flow = cls.create_flow(**kwargs)

        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }

        # acoustic model + lexicon for the flow nodes
        for node in dump_flow.get_node_names_by_filter("speech-alignment-dump"):
            mapping["lexicon"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.lexicon"
                % node
            )
            mapping["acoustic_model"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.acoustic-model"
                % node
            )

        config, post_config = rasr.build_config_from_mapping(crp, mapping, parallelize=True)

        config.acoustic_model_trainer.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = "dump.flow"
        post_config["*"].allow_overwrite = True

        dump_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, original_alignment, **kwargs):
        return dump_alignment_flow(feature_flow, original_alignment, "alignment.cache.$(TASK)")

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        dump_flow = cls.create_flow(**kwargs)
        return super().hash(
            {
                "config": config,
                "dump_flow": dump_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class AMScoresFromAlignmentLogJob(Job):
    def __init__(self, logs):
        self.logs = logs
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.out_report.get_path(), "wt") as f:
            for log in self.logs:
                if isinstance(log, dict):
                    log = log.values()
                else:
                    log = [log]

                total_score = 0.0
                total_frames = 1
                for l in log:
                    with util.uopen(tk.uncached_path(l), "rt") as infile:
                        tree = ET.parse(infile)
                    for e in tree.findall(".//alignment-statistics"):
                        total_frames += int(e.find("./frames").text)
                        total_score += float(e.find("./score/total").text)

                f.write("%f\n" % (total_score / total_frames))


class ComputeTSEJob(Job):
    """
    Compute word-level TSE (measured in frames) of some alignment. This is the average shift of word-start and word-end
    times compared to a reference. The word-start is set at the first frame which contains an allophone belonging to the
    word and word-end is the last frame. Silence/blank is ignored.

    See also https://arxiv.org/abs/2210.09951
    """

    def __init__(
        self,
        alignment_cache: tk.Path,
        ref_alignment_cache: tk.Path,
        allophone_file: tk.Path,
        ref_allophone_file: tk.Path,
        silence_phone: str = "[SILENCE]",
        ref_silence_phone: str = "[SILENCE]",
        upsample_factor: int = 1,
        ref_upsample_factor: int = 1,
        seq_tag_transform: Optional[Callable[[str], str]] = None,
        remove_outlier_limit: Optional[int] = None,
    ) -> None:
        """
        :param alignment_cache: RASR alignment cache file or bundle for which to compute TSEs
        :param ref_alignment_cache: Reference RASR alignment cache file to compare word boundaries to
        :param allophone_file: Allophone file corresponding to `alignment_cache`
        :param ref_allophone_file: Allophone file corresponding to `ref_alignment_cache`
        :param silence_phone: Silence phoneme string in lexicon corresponding to `allophone_file`
        :param ref_silence_phone: Silence phoneme string in lexicon corresponding to `ref_allophone_file`
        :param upsample_factor: Factor to upsample alignment if it was generated by a model with subsampling
        :param ref_upsample_factor: Factor to upsample reference alignment if it was generated by a model with subsampling
        :param seq_tag_transform: Function that transforms seq tag in alignment cache such that it matches the seq tags in the reference
        :param remove_outlier_limit: If set, boundary differences greater than this frame limit are discarded from computation
        """
        self.alignment_cache = alignment_cache
        self.allophone_file = allophone_file
        self.silence_phone = silence_phone
        self.upsample_factor = upsample_factor

        self.ref_alignment_cache = ref_alignment_cache
        self.ref_allophone_file = ref_allophone_file
        self.ref_silence_phone = ref_silence_phone
        self.ref_upsample_factor = ref_upsample_factor

        self.seq_tag_transform = seq_tag_transform
        self.remove_outlier_limit = remove_outlier_limit or float("inf")

        self.out_tse_frames = self.output_var("tse_frames")
        self.out_word_start_frame_differences = self.output_var("start_frame_differences")
        self.out_plot_word_start_frame_differences = self.output_path("start_frame_differences.png")
        self.out_word_end_frame_differences = self.output_var("end_frame_differences")
        self.out_plot_word_end_frame_differences = self.output_path("end_frame_differences.png")
        self.out_boundary_frame_differences = self.output_var("boundary_frame_differences")
        self.out_plot_boundary_frame_differences = self.output_path("boundary_frame_differences.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)
        yield Task("plot", resume="plot", mini_task=True)

    @staticmethod
    def _compute_word_boundaries(
        alignments: Union[rasr_cache.FileArchive, rasr_cache.FileArchiveBundle],
        allophone_map: List[str],
        seq_tag: str,
        silence_phone: str,
        upsample_factor: int,
    ) -> Tuple[List[int], List[int], int]:
        word_starts = []
        word_ends = []

        align_seq = alignments.read(seq_tag, "align")
        assert align_seq is not None

        seq_allophones = [allophone_map[item[1]] for item in align_seq]
        if upsample_factor > 1:
            seq_allophones = sum([[allo] * upsample_factor for allo in seq_allophones], [])

        for t, allophone in enumerate(seq_allophones):
            is_sil = silence_phone in allophone

            if not is_sil and "@i" in allophone and (t == 0 or seq_allophones[t - 1] != allophone):
                word_starts.append(t)

            if (
                not is_sil
                and "@f" in allophone
                and (t == len(seq_allophones) - 1 or seq_allophones[t + 1] != allophone)
            ):
                word_ends.append(t)

        return word_starts, word_ends, len(seq_allophones)

    def run(self) -> None:
        discarded_seqs = 0
        counted_seqs = 0

        start_differences = Counter()
        end_differences = Counter()
        differences = Counter()

        alignments = rasr_cache.open_file_archive(self.alignment_cache.get())
        alignments.setAllophones(self.allophone_file.get())
        if isinstance(alignments, rasr_cache.FileArchiveBundle):
            allophone_map = next(iter(alignments.archives.values())).allophones
        else:
            allophone_map = alignments.allophones

        ref_alignments = rasr_cache.open_file_archive(self.ref_alignment_cache.get())
        ref_alignments.setAllophones(self.ref_allophone_file.get())
        if isinstance(ref_alignments, rasr_cache.FileArchiveBundle):
            ref_allophone_map = next(iter(ref_alignments.archives.values())).allophones
        else:
            ref_allophone_map = ref_alignments.allophones

        file_list = [tag for tag in alignments.file_list() if not tag.endswith(".attribs")]

        for idx, seq_tag in enumerate(file_list, start=1):
            word_starts, word_ends, seq_length = self._compute_word_boundaries(
                alignments,
                allophone_map,
                seq_tag,
                self.silence_phone,
                self.upsample_factor,
            )
            assert len(word_starts) == len(
                word_ends
            ), f"Found different number of word starts ({len(word_starts)}) than word ends ({len(word_ends)}). Something seems to be broken."

            if self.seq_tag_transform is not None:
                ref_seq_tag = self.seq_tag_transform(seq_tag)
            else:
                ref_seq_tag = seq_tag

            ref_word_starts, ref_word_ends, ref_seq_length = self._compute_word_boundaries(
                ref_alignments,
                ref_allophone_map,
                ref_seq_tag,
                self.ref_silence_phone,
                self.ref_upsample_factor,
            )
            assert len(ref_word_starts) == len(
                ref_word_ends
            ), f"Found different number of word starts ({len(word_starts)}) than word ends ({len(word_ends)}) in reference. Something seems to be broken."

            if len(word_starts) != len(ref_word_starts):
                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}:\n    Discarded because the number of words in alignment ({len(word_starts)}) does not equal the number of words in reference ({len(ref_word_starts)})."
                )
                discarded_seqs += 1
                continue

            # Sometimes different feature extraction or subsampling may produce mismatched lengths that are different by one frame, so cut off at the shorter length
            shorter_seq_length = min(seq_length, ref_seq_length)
            word_starts = [min(start, shorter_seq_length) for start in word_starts]
            word_ends = [min(end, shorter_seq_length) for end in word_ends]
            ref_word_starts = [min(start, shorter_seq_length) for start in ref_word_starts]
            ref_word_ends = [min(end, shorter_seq_length) for end in ref_word_ends]

            seq_word_start_diffs = [start - ref_start for start, ref_start in zip(word_starts, ref_word_starts)]
            seq_word_end_diffs = [end - ref_end for end, ref_end in zip(word_ends, ref_word_ends)]

            # Optionally remove outliers
            seq_word_start_diffs = [diff for diff in seq_word_start_diffs if abs(diff) <= self.remove_outlier_limit]
            seq_word_end_diffs = [diff for diff in seq_word_end_diffs if abs(diff) <= self.remove_outlier_limit]

            seq_differences = seq_word_start_diffs + seq_word_end_diffs

            start_differences.update(seq_word_start_diffs)
            end_differences.update(seq_word_end_diffs)
            differences.update(seq_differences)

            if seq_differences:
                seq_tse = statistics.mean(abs(diff) for diff in seq_differences)

                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}):\n    Word start distances are {seq_word_start_diffs}\n    Word end distances are {seq_word_end_diffs}\n    Sequence TSE is {seq_tse} frames"
                )
                counted_seqs += 1
            else:
                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}):\n    Discarded since all distances are over the upper limit"
                )
                discarded_seqs += 1
                continue

        print(
            f"Processing finished. Computed TSE value based on {counted_seqs} sequences; {discarded_seqs} sequences were discarded."
        )

        self.out_word_start_frame_differences.set(
            {key: start_differences[key] for key in sorted(start_differences.keys())}
        )
        self.out_word_end_frame_differences.set({key: end_differences[key] for key in sorted(end_differences.keys())})
        self.out_boundary_frame_differences.set({key: differences[key] for key in sorted(differences.keys())})
        self.out_tse_frames.set(statistics.mean(abs(diff) for diff in differences.elements()))

    def plot(self):
        for descr, dict_file, plot_file in [
            (
                "start",
                self.out_word_start_frame_differences.get_path(),
                self.out_plot_word_start_frame_differences.get_path(),
            ),
            (
                "end",
                self.out_word_end_frame_differences.get_path(),
                self.out_plot_word_end_frame_differences.get_path(),
            ),
            (
                "boundary",
                self.out_boundary_frame_differences.get_path(),
                self.out_plot_boundary_frame_differences.get_path(),
            ),
        ]:
            with open(dict_file, "r") as f:
                diff_dict = eval(f.read())

            # Histogram plot with predefined buckets <-30, -30 to -21, -20 to -16, -15 to -11 etc.
            ranges = [-30, -20, -15, -10, -5, -1, 2, 6, 11, 16, 21, 31]

            range_strings = []
            range_strings.append(f"<{ranges[0]}")
            for idx in range(1, len(ranges)):
                range_strings.append(f"{ranges[idx - 1]} - {ranges[idx] - 1}")
            range_strings.append(f">{ranges[-1] - 1}")

            range_counts = [0] * (len(ranges) + 1)

            for key, count in diff_dict.items():
                idx = 0
                while idx < len(ranges) and ranges[idx] <= key:
                    idx += 1

                range_counts[idx] += count

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(range_strings, range_counts, color="skyblue")
            plt.xlabel(f"Word {descr} shift (frames)")
            plt.ylabel("Counts")
            plt.title(f"Word {descr} shift counts")
            plt.xticks(rotation=45)

            plt.savefig(plot_file)
