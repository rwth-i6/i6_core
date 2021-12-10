__all__ = ["MfccJob", "mfcc_flow"]

import copy

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def MfccJob(crp, mfcc_options=None, **kwargs):
    """
    :param rasr.crp.CommonRasrParameters crp:
    :param dict[str] mfcc_options:
    :rtype: FeatureExtractionJob
    """
    if mfcc_options is None:
        mfcc_options = {}
    else:
        mfcc_options = copy.deepcopy(mfcc_options)

    if "samples_options" not in mfcc_options:
        mfcc_options["samples_options"] = {}
    mfcc_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = mfcc_flow(**mfcc_options)

    port_name_mapping = {"features": "mfcc"}

    if "rtf" not in kwargs:
        kwargs["rtf"] = 0.1
    if "mem" not in kwargs:
        kwargs["mem"] = 2

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="MFCC",
        **kwargs
    )


def mfcc_flow(
    warping_function="mel",
    filter_width=268.258,
    normalize=True,
    normalization_options=None,
    without_samples=False,
    samples_options=None,
    fft_options=None,
    cepstrum_options=None,
    add_features_output=False,
):
    """
    :param warping_function str:
    :param filter_width float:
    :param normalize bool: whether to add or not a normalization layer
    :param without_samples bool:
    :param samples_options dict: arguments to :func:`~features.common.sample_flow`
    :param fft_options dict: arguments to :func:`~features.common.fft_flow`
    :param cepstrum_options dict: arguments to :func:`~features.common.cepstrum_flow`
    :param add_features_output bool: Add the output port "features" when normalize is True. This should be set to True,
        default is False to not break existing hash.
    """
    if normalization_options is None:
        normalization_options = {}
    if samples_options is None:
        samples_options = {}
    if fft_options is None:
        fft_options = {}
    if cepstrum_options is None:
        cepstrum_options = {}

    if normalize and "normalize" not in cepstrum_options:
        cepstrum_options["normalize"] = False

    net = rasr.FlowNetwork()

    if without_samples:
        net.add_input("samples")
    else:
        samples_net = samples_flow(**samples_options)
        samples_mapping = net.add_net(samples_net)

    fft_net = fft_flow(**fft_options)
    fft_mapping = net.add_net(fft_net)

    if without_samples:
        net.interconnect_inputs(fft_net, fft_mapping)
    else:
        net.interconnect(samples_net, samples_mapping, fft_net, fft_mapping)

    filterbank = net.add_node(
        "signal-filterbank",
        "filterbank",
        {"warping-function": warping_function, "filter-width": filter_width},
    )
    net.link(
        fft_mapping[fft_net.get_output_links("amplitude-spectrum").pop()], filterbank
    )

    cepstrum_net = cepstrum_flow(**cepstrum_options)
    cepstrum_mapping = net.add_net(cepstrum_net)
    for dst in cepstrum_net.get_input_links("in"):
        net.link(filterbank, cepstrum_mapping[dst])

    if normalize:
        attr = {
            "type": "mean-and-variance",
            "length": "infinity",
            "right": "infinity",
        }
        attr.update(normalization_options)
        normalization = net.add_node("signal-normalization", "mfcc-normalization", attr)
        for src in cepstrum_net.get_output_links("out"):
            net.link(cepstrum_mapping[src], normalization)
        if add_features_output:
            net.add_output("features")
        net.link(normalization, "network:features")
    else:
        net.interconnect_outputs(cepstrum_net, cepstrum_mapping, {"out": "features"})

    return net
