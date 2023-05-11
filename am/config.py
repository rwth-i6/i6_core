__all__ = ["acoustic_model_config", "get_align_config_and_crp_for_corrected_applicator"]

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

try:
    from typing import Literal
    TdpType = Union[float, Literal["infinity"]]
except ImportError:
    TdpType = Union[float, str]  # str must be: "infinity"

import i6_core.rasr as rasr

from sisyphus import tk




@dataclass
class TdpValues:
    loop: TdpType
    forward: TdpType
    skip: TdpType
    exit: TdpType


def acoustic_model_config(
    state_tying: str = "monophone",
    states_per_phone: int = 3,
    state_repetitions: int = 1,
    across_word_model: bool = True,
    early_recombination: bool = False,
    tdp_scale: float = 1.0,
    tdp_transition: Union[TdpValues, Tuple[TdpType, ...]] = (3.0, 0.0, 3.0, 2.0),
    tdp_silence: Union[TdpValues, Tuple[TdpType, ...]] = (0.0, 3.0, "infinity", 6.0),
    tying_type: Literal["global", "global-and-nonword"] = "global",
    nonword_phones: Union[str, List[str]] = "",
    tdp_nonword: Union[TdpValues, Tuple[TdpType, ...]] = (0.0, 3.0, "infinity", 6.0),
    state_tying_file: Optional[tk.Path] = None,
    phon_history_length: int = 1,
    phon_future_length: int = 1,
) -> rasr.RasrConfig:
    """
    Create a RasrConfig object with common default values to be used as `acoustic_model_config`.

    :param state_tying: Choice of state-tying type.
    :param states_per_phone: Number of states per phoneme.
    :param state_repetitions: Number of repetitions per state.
    :param across_word_model: Enable co-articulation across word boundaries.
    :param early_recombination: Enable recombination of word hypothesis before actual word end.
    :param tdp_scale: Global scaling factor that is multiplied to all tdp values.
    :param tdp_transition: tdp values assigned to transitions of regular states.
    :param tdp_silence: tdp values assigned to transitions of the silence state.
    :param tying_type: Choice of tying type for transition model.
    :param nonword_phones: Nonword-phoneme(s), e.g. [NOISE]. They get extra tdp's if tying_tpe is "global-and-nonword".
    :param tdp_nonword: tdp values assigned to transitions of nonword states. Only applied it tying_type is "global-and-nonword".
    :param state_tying_file: File containing state-tying info for e.g. `cart` or `lookup` state-tying.
    :param phon_history_length: maximum number of history tokens considered for allophone alphabet.
    :param phon_future_length: maximum number of future tokens considered for allophone alphabet.

    :return: RasrConfig using the specified values.
    """
    config = rasr.RasrConfig()

    config.state_tying.type = state_tying
    if state_tying_file is not None:
        config.state_tying.file = state_tying_file
    config.allophones.add_from_lexicon = True
    config.allophones.add_all = False

    config.hmm.states_per_phone = states_per_phone
    config.hmm.state_repetitions = state_repetitions
    config.hmm.across_word_model = across_word_model
    config.hmm.early_recombination = early_recombination

    config.tdp.scale = tdp_scale

    if not isinstance(tdp_transition, TdpValues):
        tdp_transition = TdpValues(*tdp_transition)
    config.tdp["*"].loop = tdp_transition.loop
    config.tdp["*"].forward = tdp_transition.forward
    config.tdp["*"].skip = tdp_transition.skip
    config.tdp["*"].exit = tdp_transition.exit

    if not isinstance(tdp_silence, TdpValues):
        tdp_silence = TdpValues(*tdp_silence)
    config.tdp.silence.loop = tdp_silence.loop
    config.tdp.silence.forward = tdp_silence.forward
    config.tdp.silence.skip = tdp_silence.skip
    config.tdp.silence.exit = tdp_silence.exit

    config.tdp["entry-m1"].loop = "infinity"
    config.tdp["entry-m2"].loop = "infinity"

    if tying_type == "global-and-nonword":
        config.tdp.tying_type = "global-and-nonword"
        config.tdp.nonword_phones = nonword_phones
        if not isinstance(tdp_nonword, TdpValues):
            tdp_nonword = TdpValues(*tdp_nonword)
        for nw in [0, 1]:
            k = f"nonword-{nw}"
            config.tdp[k].loop = tdp_nonword.loop
            config.tdp[k].forward = tdp_nonword.forward
            config.tdp[k].skip = tdp_nonword.skip
            config.tdp[k].exit = tdp_nonword.exit

    if phon_history_length != 1:
        config.phonology.history_length = phon_history_length
    if phon_future_length != 1:
        config.phonology.future_length = phon_future_length

    return config


def get_align_config_and_crp_for_corrected_applicator(
    crp: rasr.CommonRasrParameters, exit_penalty: float = 0.0
) -> [rasr.CommonRasrParameters, rasr.RasrConfig]:
    """
    Set the correct type of applicator, default is "legacy". Moreover, set exit penalities to zero
    For a given word sequence the exit penalty is constant with respect to the max/sum
    :param crp:
    :param exit_penalty:
    :return:
    """

    align_crp = rasr.CommonRasrParameters(base=crp)
    align_crp.acoustic_model_config.tdp.applicator_type = "corrected"
    transition_types = ["*", "silence"]
    if align_crp.acoustic_model_config.tdp.tying_type == "global-and-nonword":
        for nw in [0, 1]:
            transition_types.append(f"nonword-{nw}")
    for t in transition_types:
        align_crp.acoustic_model_config.tdp[t].exit = exit_penalty

    align_crp.acoustic_model_config.fix_allophone_context_at_word_boundaries = True
    align_crp.acoustic_model_config.transducer_builder_filter_out_invalid_allophones = True

    pre_pattern = "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.allophone-state-graph-builder.orthographic-parser"
    extra_config = rasr.RasrConfig()
    extra_config[f"{pre_pattern}.normalize-lemma-sequence-scores"] = False
    extra_config[f"{pre_pattern}.allow-for-silence-repetitions"] = False

    return align_crp, extra_config
