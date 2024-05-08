from typing import Tuple, Optional, Dict
import itertools
from dataclasses import dataclass

import numpy as np

from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchive
from sisyphus import Job, Task, tk


@dataclass
class DenseLabelInfo:
    """
    Attributes:
        n_contexts: number of phonemes in lexicon ( usually need to + 1 for non-context # in rasr)
        use_word_end_classes: if word end class is used for no tying dense label
        use_boundary_classes: if bounary class is used for no tying dense label
        num_hmm_states_per_phon: the number of hmm states per phoneme
    """

    n_contexts: int
    use_word_end_classes: bool
    use_boundary_classes: bool
    num_hmm_states_per_phon: int


class GetPhonemeLabelsFromNoTyingDense(Job):
    def __init__(
        self,
        alignment_cache_path: tk.Path,
        allophone_path: tk.Path,
        dense_tying_path: tk.Path,
        dense_label_info: DenseLabelInfo,
        sparse: bool = False,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        Get past/center/future context label of alignment by calculating back labels from dense tying and write the
        labels into hdf file.
        (C.f. NoStateTyingDense in rasr
        https://github.com/rwth-i6/rasr/blob/a942e3940c30eeba900c873f3bfb3f48d5b39ddb/src/Am/ClassicStateTying.cc#L272)

        :param alignment_cache_path: path to alginment cache
        :param allophone_path: path to allohone
        :param dense_tying_path: path to denser tying file
        :param dense_label_info: the dense label information
        :param sparse: writes the data to hdf in sparse format
        :param returnn_root: path to returnn root
        """
        self.alignment_cache_path = alignment_cache_path
        self.allophone_path = allophone_path
        self.dense_tying_path = dense_tying_path
        assert not (
            dense_label_info.use_boundary_classes and dense_label_info.use_word_end_classes
        ), "we do not use both class distinctions"
        self.dense_label_info = dense_label_info
        self.sparse = sparse
        self.returnn_root = returnn_root

        self.out_hdf_left_context = self.output_path("left_context.hdf")
        self.out_hdf_right_context = self.output_path("right_context.hdf")
        self.out_hdf_center_context = self.output_path("center_context.hdf")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def get_tying(cls, dense_tying_path: tk.Path) -> Dict[str, int]:
        """
        reads state tying file

        :return: dict state repr -> state idx
        """
        with open(dense_tying_path.get_path()) as dense_tying_file:
            state_tying = {
                k: int(v) for l in dense_tying_file for k, v in [l.strip().split()[0:2]] if not l.startswith("#")
            }

        return state_tying

    @classmethod
    def get_target_labels_from_dense(cls, dense_label: int, dense_label_info: DenseLabelInfo) -> Tuple[int, int, int]:
        num_boundary_classes = 4
        num_word_end_classes = 2

        future_label = dense_label % dense_label_info.n_contexts
        pop_future_label = dense_label // dense_label_info.n_contexts

        past_label = pop_future_label % dense_label_info.n_contexts
        center_state = pop_future_label // dense_label_info.n_contexts

        if dense_label_info.use_word_end_classes:
            word_end_class = center_state % num_word_end_classes
            center_state = center_state // num_word_end_classes

        if dense_label_info.use_boundary_classes:
            boundary_class = center_state % num_boundary_classes
            center_state = center_state // num_boundary_classes

        hmm_state_class = center_state % dense_label_info.num_hmm_states_per_phon
        center_label = center_state // dense_label_info.num_hmm_states_per_phon

        return future_label, center_label, past_label

    @classmethod
    def sanity_check(cls, max_class_index: int, dense_label_info: DenseLabelInfo):
        # sanity check to make sure that the user is setting all values of the dense tying label info correct
        max_phone_idx = dense_label_info.n_contexts - 1
        max_states_idx = dense_label_info.num_hmm_states_per_phon - 1
        expected_max_class_idx = (max_phone_idx * dense_label_info.num_hmm_states_per_phon) + max_states_idx

        if dense_label_info.use_boundary_classes:
            num_boundary_classes = 4
            max_boundary_id = 3
            expected_max_class_idx *= num_boundary_classes
            expected_max_class_idx += max_boundary_id

        if dense_label_info.use_word_end_classes:
            num_word_end_classes = 2
            expected_max_class_idx *= num_word_end_classes
            max_word_end_idx = 1
            expected_max_class_idx += max_word_end_idx

        expected_max_class_idx *= dense_label_info.n_contexts
        expected_max_class_idx += max_phone_idx

        expected_max_class_idx *= dense_label_info.n_contexts
        expected_max_class_idx += max_phone_idx

        assert expected_max_class_idx == max_class_index, "something is set wrong in dense tying label info!"

    def run(self):
        returnn_root = None if self.returnn_root is None else self.returnn_root.get_path()
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)
        out_hdf_left_context = SimpleHDFWriter(
            filename=self.out_hdf_left_context,
            dim=self.dense_label_info.n_contexts if self.sparse else 1,
            ndim=1 if self.sparse else 2,
        )
        out_hdf_right_context = SimpleHDFWriter(
            filename=self.out_hdf_right_context,
            dim=self.dense_label_info.n_contexts if self.sparse else 1,
            ndim=1 if self.sparse else 2,
        )
        out_hdf_center_context = SimpleHDFWriter(
            filename=self.out_hdf_center_context,
            dim=self.dense_label_info.n_contexts if self.sparse else 1,
            ndim=1 if self.sparse else 2,
        )

        dense_tying = self.get_tying(self.dense_tying_path)
        max_class_index = max(dense_tying.values())
        self.sanity_check(max_class_index, self.dense_label_info)

        alignment_cache = FileArchive(self.alignment_cache_path)
        alignment_cache.setAllophones(self.allophone_path)

        for file in alignment_cache.ft:
            info = alignment_cache.ft[file]
            if info.name.endswith(".attribs"):
                continue

            alignment = alignment_cache.read(file, "align")
            if not len(alignment):
                continue

            aligned_allophones = ["%s.%d" % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]
            dense_targets = [dense_tying[allo] for allo in aligned_allophones]

            # optimize the calculation by grouping
            past_label_strings = []
            center_state_strings = []
            future_label_strings = []

            for k, g in itertools.groupby(dense_targets):
                seg_len = len(list(g))
                f, c, l = self.get_target_labels_from_dense(k, self.dense_label_info)

                past_label_strings = past_label_strings + [l] * seg_len
                center_state_strings = center_state_strings + [c] * seg_len
                future_label_strings = future_label_strings + [f] * seg_len

            out_hdf_left_context.insert_batch(
                inputs=np.array(past_label_strings).reshape(1, -1)
                if self.sparse
                else np.array(past_label_strings).reshape(1, -1, 1),
                seq_len=[len(past_label_strings)],
                seq_tag=[f"{info.name}"],
            )

            out_hdf_center_context.insert_batch(
                inputs=np.array(center_state_strings).reshape(1, -1)
                if self.sparse
                else np.array(center_state_strings).reshape(1, -1, 1),
                seq_len=[len(center_state_strings)],
                seq_tag=[f"{info.name}"],
            )

            out_hdf_right_context.insert_batch(
                inputs=np.array(future_label_strings).reshape(1, -1)
                if self.sparse
                else np.array(future_label_strings).reshape(1, -1, 1),
                seq_len=[len(center_state_strings)],
                seq_tag=[f"{info.name}"],
            )

        out_hdf_left_context.close()
        out_hdf_right_context.close()
        out_hdf_center_context.close()
