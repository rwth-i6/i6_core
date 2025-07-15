__all__ = [
    "AlignSplitAccumulateSequence",
    "align_then_split_and_accumulate_sequence",
    "align_and_accumulate_sequence",
    "multiple_aligns_per_split_sequence",
    "split_and_accumulate_sequence",
    "first_split_acc_then_align_split_acc",
]
import os
from typing import Union, List, Tuple

from i6_core.am.config import get_align_config_and_crp_for_corrected_applicator
from i6_core.mm.alignment import AlignmentJob, AMScoresFromAlignmentLogJob
from i6_core.mm.flow import FlowNetwork
from i6_core.mm.mixtures import EstimateMixturesJob
from i6_core.rasr import FlagDependentFlowAttribute, DiagonalMaximumScorer


class AlignSplitAccumulateSequence:
    """
    Creates a sequence of AlignmentJobs and EstimateMixturesJobs to do HMM-GMM training.
    """

    def __init__(
        self,
        crp,
        action_sequence,
        feature_flow,
        initial_mixtures=None,
        initial_alignment=None,
        parallelization="segments",
        seq_extra_args=None,
        align_extra_args=None,
        split_extra_args=None,
        accumulate_extra_args=None,
        align_extra_rqmt=None,
        split_extra_rqmt=None,
        accumulate_extra_rqmt=None,
        align_keep_values=None,
        split_keep_values=None,
        accumulate_keep_values=None,
        feature_scorer=DiagonalMaximumScorer,
        use_corrected_applicator=False,
        alias_path=None,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param list[str] action_sequence: a list actions which can be:
            - "split"
            - "accumulate"
            - "align"

            An action can be written as e.g. "align!" to indicate that this alignment output
            should be marked as output, meaning it will be stored in `self.selected_alignment`
            and get the keep_value for "selected" (or the Sisyphus default if not defined)
        :param FlowNetwork feature_flow: flow network for feature extraction
        :param initial_mixtures: initialize the sequence with given mixtures.
            Can be None only if initial_alignment is given.
        :param initial_alignment: initialize the sequence with given alignment.
            can be None only if initial_mixtures are given.
        :param str parallelization: Unused
        :param dict[int] seq_extra_args: A dict where the key is the integer index for the step id and
            values are a kwargs dict passed the respective job
        :param dict align_extra_args: dict with kwargs added to each AlignmentJob
        :param dict split_extra_args: dict with kwargs added to each split EstimateMixtureJob
        :param dict accumulate_extra_args: dict with kwargs added to each accumulate EstimateMixtureJob
        :param dict[str] align_extra_rqmt: adds the given rqmt for all AlignmentJobs.
            Be aware that values are added to existing numbers.
        :param dict[str] split_extra_rqmt: adds the given rqmt for all "split" EstimateMixtureJobs.
            Be aware that values are added to existing numbers.
        :param dict[str] accumulate_extra_rqmt: adds the given rqmt for all "accumulate" EstimateMixtureJobs.
            Be aware that values are added to existing numbers.
        :param dict align_keep_values:
            keep values for alignment jobs, which might be indexed by "default", "selected" or the action index number.
        :param dict split_keep_values:
            keep values for split jobs, which might be indexed by "default", "selected" or the action index number.
        :param dict accumulate_keep_values:
            keep values for accumulate jobs, which might be indexed by "default", "selected" or the action index number.
        :param feature_scorer: instance of rasr.feature_scorer.FeatureScorer, providing the RASR config to configure
            the feature scoring logic
        :param bool set to True if you want to avoid using the legacy FSA with incorrect silence forward probabilities.
        :param str|None alias_path: adds an alias with the action name for each job in the sequence at the
            given path
        """
        seq_extra_args = {} if seq_extra_args is None else seq_extra_args
        align_extra_args = {} if align_extra_args is None else align_extra_args
        accumulate_extra_args = {} if accumulate_extra_args is None else accumulate_extra_args
        split_extra_args = {} if split_extra_args is None else split_extra_args

        align_keep_values = {} if align_keep_values is None else align_keep_values
        split_keep_values = {} if split_keep_values is None else split_keep_values
        accumulate_keep_values = {} if accumulate_keep_values is None else accumulate_keep_values

        def update_rqmt(rqmt, extra):
            if extra is None:
                return
            for k in extra:
                if k not in rqmt:
                    rqmt[k] = 0
                rqmt[k] += extra[k]

        assert len(action_sequence) > 0
        assert parallelization in ["segments", "bundle"]
        assert initial_mixtures is not None or initial_alignment is not None
        assert action_sequence[0].startswith("align") or initial_alignment is not None
        assert (
            action_sequence[0].startswith("split")
            or action_sequence[0].startswith("accumulate")
            or initial_mixtures is not None
        )

        self.action_sequence = action_sequence
        self.all_jobs = []
        self.all_logs = []
        self.selected_alignment_jobs = []
        self.selected_mixture_jobs = []

        self.all_mixtures = []
        self.all_alignments = []
        self.selected_mixtures = []
        self.selected_alignments = []

        self.report_job = None  # type: AMScoresFromAlignmentLogJob|None

        current_alignment = initial_alignment
        current_mixtures = initial_mixtures

        for a_idx, action in enumerate(action_sequence):
            if action.startswith("align"):
                if use_corrected_applicator:
                    align_crp, align_extra_config = get_align_config_and_crp_for_corrected_applicator(crp)
                    if "extra_config" in align_extra_args:
                        align_extra_config._update(align_extra_args["extra_config"])
                    align_extra_args["extra_config"] = align_extra_config
                else:
                    align_crp = crp
                args = {
                    "crp": align_crp,
                    "feature_flow": feature_flow,
                    "feature_scorer": feature_scorer(current_mixtures),
                }

                args.update(align_extra_args)
                if a_idx in seq_extra_args:
                    args.update(seq_extra_args[a_idx])

                job = AlignmentJob(**args)
                update_rqmt(job.rqmt, align_extra_rqmt)
                if alias_path is not None:
                    job.add_alias(os.path.join(alias_path, "action_%i_align" % a_idx))
                self.all_jobs.append(job)
                self.all_logs.append(job.out_log_file)

                current_alignment = FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": job.out_alignment_path,
                        "bundle": job.out_alignment_bundle,
                    },
                )
                current_alignment.hidden_paths = job.out_single_alignment_caches
                self.all_alignments.append(current_alignment)
                if action[-1] == "!":
                    self.selected_alignment_jobs.append(job)
                    self.selected_alignments.append(current_alignment)

                if a_idx in align_keep_values:
                    job.keep_value(align_keep_values[a_idx])
                elif action[-1] == "!" and "selected" in align_keep_values:
                    job.keep_value(align_keep_values["selected"])
                elif "default" in align_keep_values:
                    job.keep_value(align_keep_values["default"])

            elif action.startswith("split") or action.startswith("accumulate"):
                split = action.startswith("split")
                args = {
                    "crp": crp,
                    "old_mixtures": current_mixtures,
                    "feature_flow": feature_flow,
                    "alignment": current_alignment,
                    "split_first": split,
                }
                args.update(split_extra_args if split else accumulate_extra_args)
                if a_idx in seq_extra_args:
                    args.update(seq_extra_args[a_idx])

                job = EstimateMixturesJob(**args)
                update_rqmt(
                    job.accumulate_rqmt,
                    split_extra_rqmt if split else accumulate_extra_rqmt,
                )
                if alias_path is not None:
                    action_name = "split" if split else "accumulate"
                    job.add_alias(os.path.join(alias_path, "action_%i_%s" % (a_idx, action_name)))
                self.all_jobs.append(job)
                self.all_logs.append(job.out_log_file)

                current_mixtures = job.out_mixtures
                self.all_mixtures.append(current_mixtures)
                if action[-1] == "!":
                    self.selected_mixture_jobs.append(job)
                    self.selected_mixtures.append(current_mixtures)

                keep_values = split_keep_values if action.startswith("split") else accumulate_keep_values
                if a_idx in keep_values:
                    job.keep_value(keep_values[a_idx])
                elif action[-1] == "!" and "selected" in keep_values:
                    job.keep_value(keep_values["selected"])
                elif "default" in keep_values:
                    job.keep_value(keep_values["default"])

            else:
                raise ValueError("Unknown action: %s" % action)

    def get_alignment_score_report(self):
        """
        :return: report .txt file path containing the alignment scores in order of the job sequence
        """
        if self.report_job is None:
            logs = []
            for job, log in zip(self.all_jobs, self.all_logs):
                if isinstance(job, AlignmentJob):
                    logs.append(log)
            self.report_job = AMScoresFromAlignmentLogJob(logs)
        return self.report_job.out_report


def align_then_split_and_accumulate_sequence(num_align, num_accumulate, mark_accumulate=True, mark_align=True):
    assert num_align > 0 and num_accumulate > 0
    acc_str = "accumulate" + ("!" if mark_accumulate else "")
    align_str = "align" + ("!" if mark_align else "")
    return ([align_str, "split"] + ["accumulate"] * (num_accumulate - 1) + [acc_str]) * num_align


def align_and_accumulate_sequence(
    num_align: int,
    num_accumulate: int,
    mark_accumulate: Union[bool, List[int]] = True,
    mark_align: Union[bool, List[Tuple[int, int]]] = True,
):
    """
    :param num_align: number full iterations (defined by number of aligns)
    :param num_accumulate: number of accumulates per align
    :param mark_accumulate: mark all accumulates as selected or 0 idx. tuple list which accums to mark
    :param mark_align: mark all aligns as selected or 0 idx. list which aligns to mark
    :return: action sequence
    :rtype: list[str]
    """
    assert num_align > 0 and num_accumulate > 0
    acc_str = "accumulate" + ("!" if mark_accumulate is True else "")
    align_str = "align" + ("!" if mark_align is True else "")
    sequence_list = ([align_str] + ["accumulate"] * (num_accumulate - 1) + [acc_str]) * num_align

    if isinstance(mark_align, list):
        for mark in mark_align:
            assert mark < num_align, "This align does not exist %d" % mark
            assert "align" in sequence_list[mark * (num_accumulate + 1)]
            sequence_list[mark * (num_accumulate + 1)] = "align!"

    if isinstance(mark_accumulate, list):
        assert all(isinstance(pair, tuple) for pair in mark_accumulate), "Input to mark needs to be Tuple"
        for align_i, accum_i in mark_accumulate:
            assert align_i < num_align, "This align does not exist %d" % align_i
            assert accum_i < num_accumulate, "Accum sequence is shorter than requested mark %d" % accum_i
            assert "accumulate" in sequence_list[(align_i * (num_accumulate + 1) + accum_i + 1)], (align_i, accum_i)
            sequence_list[(align_i * (num_accumulate + 1) + accum_i + 1)] = "accumulate!"

    return sequence_list


def multiple_aligns_per_split_sequence(num_split, num_align, num_accumulate, mark_accumulate=True, mark_align=True):
    seq = []
    for s_idx in range(num_split):
        seq.append("split")
        for aln_idx in range(num_align):
            seq.append("align" + ("!" if aln_idx == num_align - 1 and mark_align else ""))
            for acc_idx in range(num_accumulate):
                seq.append(
                    "accumulate"
                    + ("!" if acc_idx == num_accumulate - 1 and aln_idx == num_align - 1 and mark_accumulate else "")
                )
    return seq


def split_and_accumulate_sequence(num_split, num_accumulate, mark_accumulate=True):
    assert num_split > 0 and num_accumulate > 0
    acc_str = "accumulate" + ("!" if mark_accumulate else "")
    return (["split"] + ["accumulate"] * (num_accumulate - 1) + [acc_str]) * num_split


def first_split_acc_then_align_split_acc(num_split, num_align, num_accumulate, mark_accumulate=True, mark_align=True):
    assert num_split > 0 and num_accumulate > 0
    acc_str = "accumulate" + ("!" if mark_accumulate else "")
    align_str = "align" + ("!" if mark_align else "")
    return (["split"] + ["accumulate"] * (num_accumulate - 1) + [acc_str]) * num_split + (
        [align_str, "split"] + ["accumulate"] * (num_accumulate - 1) + [acc_str]
    ) * num_align
