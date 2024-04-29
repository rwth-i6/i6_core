from typing import Tuple, Optional, Dict
import itertools

import numpy as np

from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchive
from sisyphus import Job, Path, Task, tk, gs
from apptek_asr.artefacts.factory import AbstractArtefactRepository


class GetContextLabelFromDenseTyingJob(Job):
    def __init__(
        self,
        alignment_cache_path: tk.Path,
        allophone_path: tk.Path,
        dense_tying_path: tk.Path,
        n_contexts: int,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        Get past/center/future context label of alignment by calculating back labels from dense tying and write the
        labels into hdf file.
        (C.f. NoStateTyingDense in rasr
        https://github.com/rwth-i6/rasr/blob/a942e3940c30eeba900c873f3bfb3f48d5b39ddb/src/Am/ClassicStateTying.cc#L272)


        :param alignment_cache_path: path to alginment cache
        :param allophone_path: path to allohone
        :param dense_tying_path: path to denser tying file (hint: can be obatained by setting
            crp.acoustic_model_config.state_tying.type = "no-tying-dense"
            crp.acoustic_model_config.state_tying.use_boundary_classes = "no"
            crp.acoustic_model_config.state_tying.use_word_end_classes = "no")
        :param n_contexts: number of phonemes in lexicon + 1 (for non-context # in rasr)
        :param returnn_root: path to returnn root
        """
        self.alignment_cache_path = alignment_cache_path
        self.allophone_path = allophone_path
        self.dense_tying_path = dense_tying_path
        self.n_contexts = n_contexts
        self.returnn_root = returnn_root

        self.out_hdf_left_context = self.output_path("left_context.hdf")
        self.out_hdf_right_context = self.output_path("right_context.hdf")
        self.out_hdf_center_context = self.output_path("center_context.hdf")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def get_tying_and_num_classes(cls, dense_tying_path: tk.Path) -> Tuple[Dict, int]:
        num_classes = 0
        for line in open(dense_tying_path.get_path(), "rt"):
            if not line.startswith("#"):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        state_tying = dict(
            (k, int(v))
            for l in open(dense_tying_path.get_path())
            for k, v in [l.strip().split()[0:2]]
        )

        return state_tying, num_classes

    @classmethod
    def get_target_labels_from_dense(
        cls, dense_label: int, hmm_state: int, n_contexts: int
    ) -> Tuple[int, int, int]:
        """ """

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)
        centerState = np.floor_divide(centerState - hmm_state, 3)

        return futureLabel, centerState, pastLabel

    def run(self):
        returnn_root = (
            None if self.returnn_root is None else self.returnn_root.get_path()
        )
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)
        out_hdf_left_context = SimpleHDFWriter(
            filename=self.out_hdf_left_context, dim=1
        )
        out_hdf_right_context = SimpleHDFWriter(
            filename=self.out_hdf_right_context, dim=1
        )
        out_hdf_center_context = SimpleHDFWriter(
            filename=self.out_hdf_center_context, dim=1
        )

        dense_tying, _ = self.get_tying_and_num_classes(self.dense_tying_path)

        alignment_cache = FileArchive(self.alignment_cache_path)
        alignment_cache.setAllophones(self.allophone_path)

        for file in alignment_cache.ft:
            info = alignment_cache.ft[file]
            if info.name.endswith(".attribs"):
                continue

            alignment = alignment_cache.read(file, "align")
            aligned_allophones = [
                "%s.%d" % (alignment_cache.allophones[t[1]], t[2]) for t in alignment
            ]
            dense_targets = [dense_tying[allo] for allo in aligned_allophones]
            hmm_state_ids = [alignment[i][2] for i in range(len(alignment))]

            # optimize the calculation by grouping
            pastLabel_strings = []
            centerState_strings = []
            futureLabel_strings = []

            for k, g in itertools.groupby(zip(dense_targets, hmm_state_ids)):
                segLen = len(list(g))
                dense_target, hmm_state = k
                f, c, l = self.get_target_labels_from_dense(
                    dense_target, hmm_state, self.n_contexts
                )

                pastLabel_strings = pastLabel_strings + [l] * segLen
                centerState_strings = centerState_strings + [c] * segLen
                futureLabel_strings = futureLabel_strings + [f] * segLen

            out_hdf_left_context.insert_batch(
                inputs=np.array(pastLabel_strings).reshape(1, -1, 1),
                seq_len=[len(pastLabel_strings)],
                seq_tag=[f"{info.name}"],
            )

            out_hdf_center_context.insert_batch(
                inputs=np.array(centerState_strings).reshape(1, -1, 1),
                seq_len=[len(centerState_strings)],
                seq_tag=[f"{info.name}"],
            )

            out_hdf_right_context.insert_batch(
                inputs=np.array(futureLabel_strings).reshape(1, -1, 1),
                seq_len=[len(centerState_strings)],
                seq_tag=[f"{info.name}"],
            )

        out_hdf_left_context.close()
        out_hdf_right_context.close()
        out_hdf_center_context.close()


def py():
    aar = AbstractArtefactRepository()
    runtime_name = "ApptekCluster-ubuntu2204-tf2.13.0-2023-08-25"
    runtime = aar.get_artefact_factory("runtime", runtime_name).build()
    gs.worker_wrapper = runtime.worker_wrapper

    alignment_cache_path = tk.Path(
        "/nas/data/speech/ES_US/8kHz/NameAddr/corpus/batch.1.v1/gmm_sbw_8kHz_20230621.alignment.split-1/NameAddr-batch.1.v1.alignment.cache.1"
    )
    allophone_path = tk.Path(
        "/nas/models/asr/artefacts/allophones/ES/8kHz/20230511-gmm-sbw/allophones"
    )
    dense_tying_path = tk.Path(
        "/nas/models/asr/jxu/setups/2024-04-22--jxu-multitask-left-right-center-state/work/i6_core/lexicon/allophones/DumpStateTyingJob.D6srwC6nq7bm/output/state-tying"
    )
    n_contexts = 34
    returnn_root = tk.Path(
        "/nas/models/asr/jxu/setups/2024-04-22--jxu-multitask-left-right-center-state/tools/returnn"
    )

    get_context_job = GetContextLabelFromDenseTyingJob(
        alignment_cache_path=alignment_cache_path,
        allophone_path=allophone_path,
        dense_tying_path=dense_tying_path,
        n_contexts=n_contexts,
        returnn_root=returnn_root,
    )

    tk.register_output("out_hdf_left_context", get_context_job.out_hdf_left_context)
    tk.register_output("out_hdf_right_context", get_context_job.out_hdf_right_context)
    tk.register_output("out_hdf_center_context", get_context_job.out_hdf_center_context)
