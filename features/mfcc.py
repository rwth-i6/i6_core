__all__ = ["MfccJob", "mfcc_flow"]

import copy

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def MfccJob(crp, mfcc_options=None, extra_config=None, extra_post_config=None):
    """
    :param rasr.crp.CommonRasrParameters crp:
    :param dict[str] mfcc_options:
    :param rasr.config.RasrConfig|None extra_config:
    :param rasr.config.RasrConfig|None extra_post_config:
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

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="MFCC",
        rtf=0.1,
        mem=2,
        extra_config=extra_config,
        extra_post_config=extra_post_config,
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
):
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
        net.link(normalization, "network:features")
    else:
        net.interconnect_outputs(cepstrum_net, cepstrum_mapping, {"out": "features"})

    return net
