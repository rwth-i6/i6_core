from __future__ import annotations

__all__ = ["System", "select_element", "CorpusObject"]

import copy
import types
from typing import Any, Dict, List, Optional, Tuple, Union
import re

from sisyphus import *

import i6_core.corpus as corpus_recipes
import i6_core.lexicon as lexicon
import i6_core.returnn as returnn
import i6_core.features as features
import i6_core.mm as mm
import i6_core.recognition as recog
import i6_core.rasr as rasr
import i6_core.util as util
import i6_core.vtln as vtln

from .mm_sequence import AlignSplitAccumulateSequence

Path = setup_path(__package__)


selector_type = Union[str, List, Tuple[str], Tuple[str, str], Tuple[str, str, int]]


class CorpusObject(tk.Object):
    """
    A simple container object to track additional information for a bliss corpus
    """

    def __init__(self):
        self.corpus_file: Optional[tk.Path] = None  # bliss corpus xml
        self.audio_dir: Optional[tk.Path] = None  # audio directory if paths are relative (usually not needed)
        self.audio_format: Optional[str] = None  # format type of the audio files, see e.g. get_input_node_type()
        self.duration: Optional[float] = None  # duration of the corpus, is used to determine job time


class System:
    """
    A template class to define and manage typical elements of a full RASR-based system pipeline
    """

    def __init__(self):
        self.crp = {"base": rasr.CommonRasrParameters()}
        rasr.crp_add_default_output(self.crp["base"])

        self.default_mixture_scorer = rasr.DiagonalMaximumScorer

        # collections which are a nested dict of corpus_key -> name -> some object / some container
        # the container can be list, tuple or dict
        self.alignments = {}  # type: Dict[str, Dict[str, Union[List[Any], Any]]]
        # corpus_key -> alignment_name -> element or list of e.g. FlowAttributes
        # with cache_mode containing caches and bundle
        self.ctm_files = {}
        self.feature_caches = {}
        self.feature_bundles = {}
        self.feature_flows = {}
        self.feature_scorers = {}
        self.lattice_caches = {}
        self.lattice_bundles = {}
        self.mixtures = {}
        self.nn_configs = {}
        self.nn_models = {}

        self.allophone_files = {}
        self.cart_questions = {}
        self.normalization_matrices = {}
        self.stm_files = {}
        self.glm_files = {}

        self.scorers = {}
        self.scorer_args = {}
        self.scorer_hyp_arg = {}

        self.jobs: Dict[str, Dict[str, Union[Job, AlignSplitAccumulateSequence]]] = {"base": {}}

    def set_corpus(
        self, name: str, corpus: CorpusObject, concurrent: int, segment_path: Optional[util.MultiOutputPath] = None
    ):
        """
        Initialize collections and crp for a new corpus

        :param name: will be the corpus_key
        :param corpus:
        :param concurrent:
        :param segment_path:
        :return:
        """
        self.crp[name] = rasr.CommonRasrParameters(base=self.crp["base"])
        rasr.crp_set_corpus(self.crp[name], corpus)
        self.crp[name].concurrent = concurrent
        self.crp[name].segment_path = segment_path

        self.alignments[name] = {}
        self.ctm_files[name] = {}
        self.feature_caches[name] = {}
        self.feature_bundles[name] = {}
        self.feature_flows[name] = {}
        self.feature_scorers[name] = {}
        self.lattice_caches[name] = {}
        self.lattice_bundles[name] = {}
        self.mixtures[name] = {}
        self.nn_configs[name] = {}
        self.nn_models[name] = {}

        self.normalization_matrices[name] = {}

        self.jobs[name] = {}

    def add_overlay(self, origin: str, name: str):
        """
        Creates an overlay (meaning e.g. a subset or other kind of modified version) of an existing corpus

        :param origin: name/corpus_key of the original common rasr parameters with associated alignments, features, etc
        :param name: name/corpus_key of the new overlay over the original
        :return:
        """
        self.crp[name] = rasr.CommonRasrParameters(base=self.crp[origin])

        self.alignments[name] = {}
        self.ctm_files[name] = {}
        self.feature_caches[name] = copy.deepcopy(self.feature_caches[origin])
        self.feature_bundles[name] = copy.deepcopy(self.feature_bundles[origin])
        self.feature_flows[name] = copy.deepcopy(self.feature_flows[origin])
        self.feature_scorers[name] = copy.deepcopy(self.feature_scorers[origin])
        self.lattice_caches[name] = {}
        self.lattice_bundles[name] = {}
        self.mixtures[name] = {}
        self.nn_configs[name] = {}
        self.nn_models[name] = {}

        self.normalization_matrices[name] = {}

        self.jobs[name] = {}

    def copy_from_system(self, origin_system: System, origin_name: str, target_name: Optional[str] = None):
        """
        Import the dictionaries from another System

        :param origin_system:
        :param origin_name: Name of the dataset in another system
        :param target_name: Name of the dataset in current system (optional) (default=origin_name)
        :return:
        """
        if not target_name:
            target_name = origin_name

        self.crp[target_name] = rasr.CommonRasrParameters(base=origin_system.crp[origin_name])

        self.alignments[target_name] = copy.deepcopy(origin_system.alignments[origin_name])
        self.ctm_files[target_name] = copy.deepcopy(origin_system.ctm_files[origin_name])
        self.feature_caches[target_name] = copy.deepcopy(origin_system.feature_caches[origin_name])
        self.feature_bundles[target_name] = copy.deepcopy(origin_system.feature_bundles[origin_name])
        self.feature_flows[target_name] = copy.deepcopy(origin_system.feature_flows[origin_name])
        self.feature_scorers[target_name] = copy.deepcopy(origin_system.feature_scorers[origin_name])
        self.lattice_caches[target_name] = copy.deepcopy(origin_system.lattice_caches[origin_name])
        self.lattice_bundles[target_name] = copy.deepcopy(origin_system.lattice_bundles[origin_name])
        self.mixtures[target_name] = copy.deepcopy(origin_system.mixtures[origin_name])
        self.nn_configs[target_name] = copy.deepcopy(origin_system.nn_configs[origin_name])
        self.nn_models[target_name] = copy.deepcopy(origin_system.nn_models[origin_name])
        # self.stm_files      [target_name] = copy.deepcopy(origin_system.stm_files[origin_name])

        self.normalization_matrices[target_name] = copy.deepcopy(origin_system.normalization_matrices[origin_name])

        self.jobs[target_name] = {}

    def set_sclite_scorer(self, corpus: str, **kwargs):
        """
        set sclite scorer for a corpus

        :param corpus: corpus name
        :param kwargs:
        :return:
        """
        self.scorers[corpus] = recog.ScliteJob
        self.scorer_args[corpus] = {"ref": self.stm_files[corpus], **kwargs}
        self.scorer_hyp_arg[corpus] = "hyp"

    def set_hub5_scorer(self, corpus: str, **kwargs):
        """
        set hub5 scorer for a corpus

        :param corpus: corpus name
        :param kwargs:
        :return:
        """
        self.scorers[corpus] = recog.Hub5ScoreJob
        assert corpus in self.glm_files, (
            "No glm file was defined for '%s' corpus, please specify it explicitly. "
            "For all non-inhouse corpora there should be an official glm file." % corpus
        )
        self.scorer_args[corpus] = {
            "ref": self.stm_files[corpus],
            "glm": self.glm_files[corpus],
            **kwargs,
        }
        self.scorer_hyp_arg[corpus] = "hyp"

    def set_kaldi_scorer(self, corpus: str, mapping: Dict):
        """
        set kaldi scorer for a corpus

        :param corpus: corpus name
        :param mapping: dictionary of words to be replaced
        :return:
        """
        self.scorers[corpus] = recog.KaldiScorerJob
        self.scorer_args[corpus] = {
            "corpus_path": self.crp[corpus].corpus_config.file,
            "map": mapping,
        }
        self.scorer_hyp_arg[corpus] = "ctm"

    def create_stm_from_corpus(self, corpus: str, **kwargs):
        """
        create and set the stm files

        :param corpus: corpus name
        :param kwargs: additional arguments for the CorpusToStmJob
        :return:
        """
        self.stm_files[corpus] = corpus_recipes.CorpusToStmJob(
            self.crp[corpus].corpus_config.file, **kwargs
        ).out_stm_path

    def store_allophones(self, source_corpus: str, target_corpus: str = "base", **kwargs):
        """
        dump allophones into a file

        :param source_corpus:
        :param target_corpus:
        :param kwargs:
        :return:
        """
        self.jobs[target_corpus]["allophones"] = lexicon.StoreAllophonesJob(self.crp[source_corpus], **kwargs)
        # noinspection PyUnresolvedReferences
        self.allophone_files[target_corpus] = self.jobs[target_corpus]["allophones"].out_allophone_file
        if self.crp[target_corpus].acoustic_model_post_config is None:
            self.crp[target_corpus].acoustic_model_post_config = rasr.RasrConfig()
        self.crp[target_corpus].acoustic_model_post_config.allophones.add_from_file = self.allophone_files[
            target_corpus
        ]

    def replace_named_flow_attr(self, corpus: str, flow_regex: str, attr_name: str, new_value: Any):
        """

        :param corpus:
        :param flow_regex:
        :param attr_name:
        :param new_value:
        :return:
        """
        pattern = re.compile(flow_regex)
        for name, flow in self.feature_flows[corpus].items():
            if pattern.match(name) and attr_name in flow.named_attributes:
                flow.named_attributes[attr_name].value = new_value

    def add_derivatives(self, corpus: str, flow: str, num_deriv: int, num_features: Optional[int] = None):
        """
        :param corpus:
        :param flow:
        :param num_deriv:
        :param num_features:
        :return:
        """
        self.feature_flows[corpus][flow + "+deriv"] = features.add_derivatives(
            self.feature_flows[corpus][flow], num_deriv
        )
        if num_features is not None:
            self.feature_flows[corpus][flow + "+deriv"] = features.select_features(
                self.feature_flows[corpus][flow + "+deriv"], "0-%d" % (num_features - 1)
            )

    def energy_features(self, corpus: str, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["energy_features"] = f = features.EnergyJob(self.crp[corpus], **kwargs)
        f.add_alias("%s%s_energy_features" % (prefix, corpus))
        self.feature_caches[corpus]["energy"] = f.out_feature_path["energy"]
        self.feature_bundles[corpus]["energy"] = f.out_feature_bundle["energy"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["energy"],
                "bundle": self.feature_bundles[corpus]["energy"],
            },
        )
        self.feature_flows[corpus]["energy"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_energy"] = f.feature_flow

    def mfcc_features(self, corpus: str, num_deriv: int = 2, num_features: int = 33, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param num_deriv:
        :param num_features:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["mfcc_features"] = f = features.MfccJob(self.crp[corpus], **kwargs)
        f.add_alias("%s%s_mfcc_features" % (prefix, corpus))
        self.feature_caches[corpus]["mfcc"] = f.out_feature_path["mfcc"]
        self.feature_bundles[corpus]["mfcc"] = f.out_feature_bundle["mfcc"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["mfcc"],
                "bundle": self.feature_bundles[corpus]["mfcc"],
            },
        )
        self.feature_flows[corpus]["mfcc"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_mfcc"] = f.feature_flow
        self.add_derivatives(corpus, "mfcc", num_deriv, num_features)
        self.add_derivatives(corpus, "uncached_mfcc", num_deriv, num_features)

    def fb_features(self, corpus: str, **kwargs):
        """
        :param corpus:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["fb_features"] = f = features.FilterbankJob(self.crp[corpus], **kwargs)
        f.add_alias("%s_fb_features" % corpus)
        self.feature_caches[corpus]["fb"] = f.out_feature_path["fb"]
        self.feature_bundles[corpus]["fb"] = f.out_feature_bundle["fb"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["fb"],
                "bundle": self.feature_bundles[corpus]["fb"],
            },
        )
        self.feature_flows[corpus]["fb"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_fb"] = f.feature_flow

    def gt_features(self, corpus: str, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["gt_features"] = f = features.GammatoneJob(self.crp[corpus], **kwargs)
        if "gt_options" in kwargs and "channels" in kwargs.get("gt_options"):
            f.add_alias("%s%s_gt_%i_features" % (prefix, corpus, kwargs.get("gt_options").get("channels")))
        else:
            f.add_alias("%s%s_gt_features" % (prefix, corpus))
        self.feature_caches[corpus]["gt"] = f.out_feature_path["gt"]
        self.feature_bundles[corpus]["gt"] = f.out_feature_bundle["gt"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["gt"],
                "bundle": self.feature_bundles[corpus]["gt"],
            },
        )
        self.feature_flows[corpus]["gt"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_gt"] = f.feature_flow

    def generic_features(
        self,
        corpus: str,
        name: str,
        feature_flow: rasr.FlowNetwork,
        port_name: str = "features",
        prefix: str = "",
        **kwargs,
    ):
        """
        :param corpus: corpus identifier
        :param name: feature identifier, like "mfcc". Also used in the naming of the output feature caches.
        :param feature_flow: definition of the RASR feature flow network
        :param port_name: output port of the flow network to use, e.g., for samples_flow() this is "samples"
        :param prefix: prefix for the alias job symlink
        :param kwargs:
        :return:
        """
        assert (
            port_name in feature_flow.outputs
        ), f"port name '{port_name}' not available in feature flow (available are: {feature_flow.outputs})"
        port_name_mapping = {port_name: name}
        self.jobs[corpus][f"{name}_features"] = f = features.FeatureExtractionJob(
            self.crp[corpus], feature_flow, port_name_mapping, job_name=name, **kwargs
        )
        f.add_alias(f"{prefix}{corpus}_{name}_features")
        self.feature_caches[corpus][name] = f.out_feature_path[name]
        self.feature_bundles[corpus][name] = f.out_feature_bundle[name]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus][name],
                "bundle": self.feature_bundles[corpus][name],
            },
        )
        self.feature_flows[corpus][name] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus][f"uncached_{name}"] = f.feature_flow

    def plp_features(self, corpus: str, num_deriv: int = 2, num_features: int = 23, **kwargs):
        """
        :param corpus:
        :param num_deriv:
        :param num_features:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["plp_features"] = f = features.PlpJob(self.crp[corpus], **kwargs)
        f.add_alias("%s_plp_features" % corpus)
        self.feature_caches[corpus]["plp"] = f.out_feature_path["plp"]
        self.feature_bundles[corpus]["plp"] = f.out_feature_bundle["plp"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["plp"],
                "bundle": self.feature_bundles[corpus]["plp"],
            },
        )
        self.feature_flows[corpus]["plp"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["plp+deriv"] = features.add_derivatives(self.feature_flows[corpus]["plp"], num_deriv)
        if num_features is not None:
            self.feature_flows[corpus]["plp+deriv"] = features.select_features(
                self.feature_flows[corpus]["plp+deriv"], "0-%d" % (num_features - 1)
            )
        self.feature_flows[corpus]["uncached_plp"] = f.feature_flow

    def voiced_features(self, corpus: str, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["voiced_features"] = f = features.VoicedJob(self.crp[corpus], **kwargs)
        f.add_alias("%s_%s_voiced_features" % (prefix, corpus))
        self.feature_caches[corpus]["voiced"] = f.out_feature_path["voiced"]
        self.feature_bundles[corpus]["voiced"] = f.out_feature_bundle["voiced"]

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["voiced"],
                "bundle": self.feature_bundles[corpus]["voiced"],
            },
        )
        self.feature_flows[corpus]["voiced"] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_voiced"] = f.feature_flow

    def tone_features(self, corpus: str, timestamp_flow: str, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param timestamp_flow:
        :param prefix:
        :param kwargs:
        :return:
        """
        timestamp_flow = self.feature_flows[corpus][timestamp_flow]
        self.jobs[corpus]["tone_features"] = f = features.ToneJob(
            self.crp[corpus], timestamp_flow=timestamp_flow, **kwargs
        )
        f.add_alias("%s%s_tone_features" % (prefix, corpus))
        self.feature_caches[corpus]["tone"] = f.out_feature_path
        self.feature_bundles[corpus]["tone"] = f.out_feature_bundle

        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus]["tone"],
                "bundle": self.feature_bundles[corpus]["tone"],
            },
        )
        self.feature_flows[corpus]["tone"] = features.basic_cache_flow(feature_path)

    def vtln_features(self, name: str, corpus: str, raw_feature_flow: rasr.FlowNetwork, warping_map: tk.Path, **kwargs):
        """
        :param name:
        :param corpus:
        :param raw_feature_flow:
        :param warping_map:
        :param kwargs:
        :return:
        """
        name = "%s+vtln" % name
        self.jobs[corpus]["%s_features" % name] = f = vtln.VTLNFeaturesJob(
            self.crp[corpus], raw_feature_flow, warping_map, **kwargs
        )
        self.feature_caches[corpus][name] = f.out_feature_path["vtln"]
        self.feature_bundles[corpus][name] = f.out_feature_bundle["vtln"]
        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus][name],
                "bundle": self.feature_bundles[corpus][name],
            },
        )
        self.feature_flows[corpus][name] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus]["uncached_" + name] = f.feature_flow

    def add_energy_to_features(self, corpus: str, flow: str):
        """
        :param corpus:
        :param flow:
        :return:
        """
        self.feature_flows[corpus]["energy,%s" % flow] = features.sync_energy_features(
            self.feature_flows[corpus][flow], self.feature_flows[corpus]["energy"]
        )

    def normalize(self, corpus: str, flow: str, target_corpora: List[str], **kwargs):
        """
        :param corpus:
        :param flow:
        :param target_corpora:
        :param kwargs:
        :return:
        """
        feature_flow = copy.deepcopy(self.feature_flows[corpus][flow])
        feature_flow.flags = {"cache_mode": "bundle"}
        new_flow_name = "%s+norm" % flow

        self.jobs[corpus]["normalize_%s" % flow] = j = features.CovarianceNormalizationJob(
            self.crp[corpus], feature_flow, **kwargs
        )
        self.normalization_matrices[corpus][new_flow_name] = j.normalization_matrix

        for c in target_corpora:
            self.feature_flows[c][new_flow_name] = features.add_linear_transform(
                self.feature_flows[c][flow], j.normalization_matrix
            )

    def costa(self, corpus: str, prefix: str = "", **kwargs):
        """
        :param corpus:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.jobs[corpus]["costa"] = j = corpus_recipes.CostaJob(self.crp[corpus], **kwargs)
        j.add_alias("%scosta_%s" % (prefix, corpus))
        tk.register_output("%s%s.costa.log.gz" % (prefix, corpus), j.out_log_file)

    def linear_alignment(self, name: str, corpus: str, flow: str, prefix: str = "", **kwargs):
        """
        :param name:
        :param corpus:
        :param flow:
        :param prefix:
        """
        name = "linear_alignment_%s" % name
        self.jobs[corpus][name] = j = mm.LinearAlignmentJob(
            self.crp[corpus], self.feature_flows[corpus][flow], **kwargs
        )
        j.add_alias(prefix + name)
        self.mixtures[corpus][name] = j.out_mixtures
        if hasattr(j, "out_alignment_path") and hasattr(j, "out_alignment_bundle"):
            self.alignments[corpus][name] = rasr.FlagDependentFlowAttribute(
                "cache_mode",
                {
                    "task_dependent": j.out_alignment_path,
                    "bundle": j.out_alignment_bundle,
                },
            )

    def align(
        self,
        name: str,
        corpus: str,
        flow: Union[selector_type, rasr.FlowNetwork],
        feature_scorer: Union[selector_type, rasr.FeatureScorer],
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param flow:
        :param feature_scorer:
        :param kwargs:
        :return:
        """
        j = mm.AlignmentJob(
            crp=self.crp[corpus],
            feature_flow=select_element(self.feature_flows, corpus, flow),
            feature_scorer=select_element(self.feature_scorers, corpus, feature_scorer),
            **kwargs,
        )

        self.jobs[corpus]["alignment_%s" % name] = j
        self.alignments[corpus][name] = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {"task_dependent": j.out_alignment_path, "bundle": j.out_alignment_bundle},
        )

    def estimate_mixtures(
        self,
        name: str,
        corpus: str,
        flow: str,
        old_mixtures: Optional[Union[selector_type, tk.Path]] = None,
        alignment: Optional[Union[selector_type, rasr.FlagDependentFlowAttribute, tk.Path]] = None,
        prefix: str = "",
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param flow:
        :param old_mixtures:
        :param alignment:
        :param prefix:
        :param kwargs:
        :return:
        """
        name = "estimate_mixtures_%s" % name
        j = mm.EstimateMixturesJob(
            crp=self.crp[corpus],
            old_mixtures=select_element(self.mixtures, corpus, old_mixtures),
            feature_flow=self.feature_flows[corpus][flow],
            alignment=select_element(self.alignments, corpus, alignment),
            **kwargs,
        )
        j.add_alias("{}{}".format(prefix, name))

        self.jobs[corpus][name] = j
        self.mixtures[corpus][name] = j.out_mixtures
        self.feature_scorers[corpus][name] = self.default_mixture_scorer(j.out_mixtures)

    def train(
        self,
        name: str,
        corpus: str,
        sequence: List[str],
        flow: Union[selector_type, rasr.FlowNetwork],
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param sequence: action sequence
        :param flow:
        :param kwargs: passed on to :class:`AlignSplitAccumulateSequence`
        """
        name = "train_%s" % name
        if "feature_scorer" not in kwargs:
            kwargs["feature_scorer"] = self.default_mixture_scorer
        self.jobs[corpus][name] = j = AlignSplitAccumulateSequence(
            self.crp[corpus],
            sequence,
            select_element(self.feature_flows, corpus, flow),
            **kwargs,
        )

        self.mixtures[corpus][name] = j.selected_mixtures
        self.alignments[corpus][name] = j.selected_alignments
        self.feature_scorers[corpus][name] = [self.default_mixture_scorer(m) for m in j.selected_mixtures]

    def recog(
        self,
        name: str,
        corpus: str,
        flow: Union[selector_type, rasr.FlowNetwork],
        feature_scorer: Union[selector_type, rasr.FeatureScorer],
        pronunciation_scale: float,
        lm_scale: float,
        parallelize_conversion: bool = False,
        lattice_to_ctm_kwargs: Optional[Dict] = None,
        prefix: str = "",
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param flow:
        :param feature_scorer:
        :param pronunciation_scale:
        :param lm_scale:
        :param parallelize_conversion:
        :param lattice_to_ctm_kwargs:
        :param prefix:
        :param kwargs:
        :return:
        """
        if lattice_to_ctm_kwargs is None:
            lattice_to_ctm_kwargs = {}

        self.crp[corpus].language_model_config.scale = lm_scale
        model_combination_config = rasr.RasrConfig()
        model_combination_config.pronunciation_scale = pronunciation_scale

        rec = recog.AdvancedTreeSearchJob(
            crp=self.crp[corpus],
            feature_flow=select_element(self.feature_flows, corpus, flow),
            feature_scorer=select_element(self.feature_scorers, corpus, feature_scorer),
            model_combination_config=model_combination_config,
            **kwargs,
        )
        rec.set_vis_name("Recog %s%s" % (prefix, name))
        rec.add_alias("%srecog_%s" % (prefix, name))
        self.jobs[corpus]["recog_%s" % name] = rec

        self.jobs[corpus]["lat2ctm_%s" % name] = lat2ctm = recog.LatticeToCtmJob(
            crp=self.crp[corpus],
            lattice_cache=rec.out_lattice_bundle,
            parallelize=parallelize_conversion,
            **lattice_to_ctm_kwargs,
        )
        lat2ctm.add_alias("%s%s_conv_to_ctm" % (prefix, name))
        self.ctm_files[corpus]["recog_%s" % name] = lat2ctm.out_ctm_file

        kwargs = copy.deepcopy(self.scorer_args[corpus])
        kwargs[self.scorer_hyp_arg[corpus]] = lat2ctm.out_ctm_file
        scorer = self.scorers[corpus](**kwargs)

        self.jobs[corpus]["scorer_%s" % name] = scorer
        tk.register_output("%srecog_%s.reports" % (prefix, name), scorer.out_report_dir)
        if scorer.out_wer:
            tk.register_output("%swers/recog_%s.wer" % (prefix, name), scorer.out_wer)

    def optimize_am_lm(
        self, name: str, corpus: str, initial_am_scale: float, initial_lm_scale: float, prefix: str = "", **kwargs
    ):
        """
        :param name:
        :param corpus:
        :param initial_am_scale:
        :param initial_lm_scale:
        :param prefix:
        :param kwargs:
        :return:
        """
        j = recog.OptimizeAMandLMScaleJob(
            crp=self.crp[corpus],
            lattice_cache=self.jobs[corpus][name].out_lattice_bundle,  # noqa, job type is not known
            initial_am_scale=initial_am_scale,
            initial_lm_scale=initial_lm_scale,
            scorer_cls=self.scorers[corpus],
            scorer_kwargs=self.scorer_args[corpus],
            scorer_hyp_param_name=self.scorer_hyp_arg[corpus],
            **kwargs,
        )
        j.add_alias("%s_optimize_am_and_lm_scale" % name)
        self.jobs[corpus]["optimize_%s" % name] = j
        tk.register_output("%soptimize_%s.log" % (prefix, name), j.out_log_file)

    def recog_and_optimize(
        self,
        name: str,
        corpus: str,
        flow: Union[selector_type, rasr.FlowNetwork],
        feature_scorer: Union[selector_type, rasr.FeatureScorer],
        pronunciation_scale: float,
        lm_scale: float,
        parallelize_conversion: bool = False,
        lattice_to_ctm_kwargs: Optional[Dict] = None,
        prefix: str = "",
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param flow:
        :param feature_scorer:
        :param pronunciation_scale:
        :param lm_scale:
        :param parallelize_conversion:
        :param lattice_to_ctm_kwargs:
        :param prefix:
        :param kwargs:
        :return:
        """
        self.recog(
            name,
            corpus,
            flow,
            feature_scorer,
            pronunciation_scale,
            lm_scale,
            parallelize_conversion,
            lattice_to_ctm_kwargs,
            prefix,
            **kwargs,
        )

        recog_name = "recog_%s" % name
        opt_name = "optimize_%s" % recog_name

        self.optimize_am_lm(
            recog_name,
            corpus,
            pronunciation_scale,
            lm_scale,
            prefix=prefix,
            opt_only_lm_scale=True,
        )

        opt_job = self.jobs[corpus][opt_name]

        self.recog(
            name + "-optlm",
            corpus,
            flow,
            feature_scorer,
            pronunciation_scale,
            opt_job.out_best_lm_score,  # noqa, job type is not known
            parallelize_conversion,
            lattice_to_ctm_kwargs,
            prefix,
            **kwargs,
        )

    def train_nn(
        self,
        name: str,
        feature_corpus: str,
        train_corpus: str,
        dev_corpus: str,
        feature_flow: str,
        alignment: Union[selector_type, rasr.FlagDependentFlowAttribute],
        returnn_config: returnn.ReturnnConfig,
        num_classes: Union[int, types.FunctionType],
        **kwargs,
    ):
        """
        :param name:
        :param feature_corpus:
        :param train_corpus:
        :param dev_corpus:
        :param feature_flow:
        :param alignment:
        :param returnn_config:
        :param num_classes:
        :param kwargs:
        :return:
        """
        assert isinstance(returnn_config, returnn.ReturnnConfig), (
            "Passing returnn_config as dict to train_nn is no longer supported, "
            "please construct a ReturnnConfig object instead"
        )
        j = returnn.ReturnnRasrTrainingJob(
            train_crp=self.crp[train_corpus],
            dev_crp=self.crp[dev_corpus],
            feature_flow=self.feature_flows[feature_corpus][feature_flow],
            alignment=select_element(self.alignments, feature_corpus, alignment),
            returnn_config=returnn_config,
            num_classes=self.functor_value(num_classes),
            **kwargs,
        )
        self.jobs[feature_corpus]["train_nn_%s" % name] = j
        self.nn_models[feature_corpus][name] = j.out_models
        self.nn_configs[feature_corpus][name] = j.out_returnn_config_file

    def create_nn_feature_scorer(
        self,
        name: str,
        corpus: str,
        feature_dimension: Union[int, types.FunctionType],
        output_dimension: Union[int, types.FunctionType],
        prior_mixtures: Union[selector_type, tk.Path],
        model: Union[selector_type, returnn.ReturnnModel],
        prior_scale: float,
        prior_file: tk.Path = None,
        **kwargs,
    ):
        """
        :param name:
        :param corpus:
        :param feature_dimension:
        :param output_dimension:
        :param prior_mixtures:
        :param model:
        :param prior_scale:
        :param prior_file:
        :param kwargs:
        :return:
        """
        scorer = rasr.ReturnnScorer(
            feature_dimension=self.functor_value(feature_dimension),
            output_dimension=self.functor_value(output_dimension),
            prior_mixtures=select_element(self.mixtures, corpus, prior_mixtures),
            model=select_element(self.nn_models, corpus, model),
            prior_scale=prior_scale,
            prior_file=prior_file,
            **kwargs,
        )
        self.feature_scorers[corpus][name] = scorer

    def functor_value(self, value):
        if isinstance(value, types.FunctionType):
            return value(self)
        else:
            return value


def select_element(collection: Dict, default_corpus: str, selector: Any, default_index: int = -1):
    """
    Select one element of the meta.System collection variables, e.g. one of:

     - System.alignments
     - System.feature_flows
     - System.feature_scorers
     - System.mixtures
     - ...

    :param collection: any meta.System collection
    :param default_corpus: a corpus key
    :param selector: some selector
    :param default_index: what element to pick if the collection is a tuple, list or dict,
        defaults to the last element (e.g. last alignment)
    :return:
    :rtype: Any
    """
    if not isinstance(selector, (list, tuple, str)):
        return selector
    else:
        if isinstance(selector, str):
            args = [default_corpus, selector, default_index]
        else:
            args = {
                1: lambda: [default_corpus, selector[0], default_index],
                2: lambda: [selector[0], selector[1], default_index],
                3: lambda: [selector[0], selector[1], selector[2]],
            }[len(selector)]()

        if default_corpus is not None:
            e = collection[args[0]]
        else:
            e = collection
        e = e[args[1]]
        if isinstance(e, (list, dict)):
            e = e[args[2]]
        return e
