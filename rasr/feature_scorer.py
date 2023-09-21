__all__ = [
    "FeatureScorer",
    "GMMFeatureScorer",
    "DiagonalMaximumScorer",
    "SimdDiagonalMaximumScorer",
    "PreselectionBatchIntScorer",
    "ReturnnScorer",
    "InvAlignmentPassThroughFeatureScorer",
    "PrecomputedHybridFeatureScorer",
    "OnnxFeatureScorer",
]

from sisyphus import *

Path = setup_path(__package__)

import os

from .config import *
from i6_core.util import get_returnn_root


class FeatureScorer:
    def __init__(self):
        self.config = RasrConfig()
        self.post_config = RasrConfig()

    def apply_config(self, path, config, post_config):
        config[path]._update(self.config)
        post_config[path]._update(self.post_config)

    def html(self):
        config = repr(self.config).replace("\n", "<br />\n")
        post_config = repr(self.post_config).replace("\n", "<br />\n")
        return "<h3>Config:</h3>\n%s<br />\n<h3>Post Config:</h3>\n%s" % (
            config,
            post_config,
        )


class GMMFeatureScorer(FeatureScorer):
    def __init__(self, mixtures, scale=1.0):
        super().__init__()
        self.config.scale = scale
        self.config.file = mixtures


class DiagonalMaximumScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "diagonal-maximum"


class SimdDiagonalMaximumScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "SIMD-diagonal-maximum"


class PreselectionBatchIntScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "preselection-batch-int"


class ReturnnScorer(FeatureScorer):
    def __init__(
        self,
        feature_dimension,
        output_dimension,
        prior_mixtures,
        model,
        mixture_scale=1.0,
        prior_scale=1.0,
        prior_file=None,
        returnn_root=None,
    ):
        super().__init__()

        self.config.feature_dimension = feature_dimension
        self.config.feature_scorer_type = "nn-trainer-feature-scorer"
        self.config.file = prior_mixtures
        self.config.priori_scale = prior_scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        else:
            self.config.normalize_mixture_weights = False
        self.config.pymod_name = "returnn.SprintInterface"
        self.config.pymod_path = get_returnn_root(returnn_root).join_right("..")
        self.config.pymod_config = StringWrapper(
            "epoch:%d,action:forward,configfile:%s" % (model.epoch, model.returnn_config_file),
            model,
        )
        self.config.scale = mixture_scale
        self.config.target_mode = "forward-only"
        self.config.trainer = "python-trainer"
        self.config.trainer_output_dimension = output_dimension
        self.config.use_network = False

        self.returnn_config = model.returnn_config_file


class InvAlignmentPassThroughFeatureScorer(FeatureScorer):
    def __init__(self, prior_mixtures, max_segment_length, mapping, priori_scale=0.0):
        super().__init__()

        self.config = RasrConfig()
        self.config.feature_scorer_type = "inv-alignment-pass-through"
        self.config.file = prior_mixtures
        self.config.max_segment_length = max_segment_length
        self.config.mapping = mapping
        self.config.priori_scale = priori_scale
        self.config.normalize_mixture_weights = False


class PrecomputedHybridFeatureScorer(FeatureScorer):
    def __init__(self, prior_mixtures, scale=1.0, priori_scale=0.0, prior_file=None):
        super().__init__()

        self.config = RasrConfig()
        self.config.feature_scorer_type = "nn-precomputed-hybrid"
        self.config.file = prior_mixtures
        self.config.scale = scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        self.config.priori_scale = priori_scale
        self.config.normalize_mixture_weights = False


class OnnxFeatureScorer(rasr.FeatureScorer):
    def __init__(
        self,
        mixtures,
        model,
        io_map,
        *args,
        label_log_posterior_scale=1.0,
        label_prior_scale=0.7,
        label_log_prior_file=None,
        apply_log_on_output=False,
        negate_output=True,
        intra_op_threads=1,
        inter_op_threads=1,
        **kwargs,
    ):
        """
        :param str mixtures: path to a *.mix file e.g. output of either EstimateMixturesJob or CreateDummyMixturesJob
        :param str model: path of a model e.g. output of ExportPyTorchModelToOnnxJob
        :param dict io_map: mapping between internal rasr identifiers and the model related input/output
        :param float label_log_posterior_scale: scales for the log probability of a label e.g. 1.0 is recommended
        :param float label_prior_scale: scale for the prior log probability of a label reasonable e.g. values in [0.1, 0.7] interval
        :param str label_log_prior_file: xml file containing log prior probabilities e.g. estimated from the model via povey method
        :param bool apply_log_on_output: whether to apply the log-function on the output, usefull if the model outputs softmax instead of log-softmax
        :param bool negate_output: wheter negate output (because the model outputs log softmax and not negative log softmax
        """
        super().__init__(*args, **kwargs)

        self.config.feature_scorer_type = "onnx-feature-scorer"
        self.config.file = mixtures
        self.config.scale = label_log_posterior_scale
        self.config.priori_scale = label_prior_scale
        if label_log_prior_file is not None:
            self.config.prior_file = label_log_prior_file

        self.config.session.file = model

        if label_log_prior_file:
            self.config.apply_log_on_output = apply_log_on_output
        if not negate_output:
            self.config.negate_output = negate_output

        self.post_config.session.intra_op_num_threads = intra_op_threads
        self.post_config.session.inter_op_num_threads = inter_op_threads

        for k, v in io_map.items():
            self.config.io_map[k] = v
