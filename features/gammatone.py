__all__ = ["GammatoneJob", "gammatone_flow"]

import copy

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def GammatoneJob(crp, gt_options=None, extra_config=None, extra_post_config=None):
    if gt_options is None:
        gt_options = {}
    else:
        gt_options = copy.deepcopy(gt_options)
    if "samples_options" not in gt_options:
        gt_options["samples_options"] = {}
    gt_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = gammatone_flow(**gt_options)

    port_name_mapping = {"features": "gt"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="Gammatone",
        rtf=0.1,
        mem=2,
        extra_config=extra_config,
        extra_post_config=extra_post_config,
    )


def gammatone_flow(
    minfreq=100,
    maxfreq=7500,
    channels=68,
    warp_freqbreak=None,
    tempint_type="hanning",
    tempint_shift=0.01,
    tempint_length=0.025,
    flush_before_gap=True,
    do_specint=True,
    specint_type="hanning",
    specint_shift=4,
    specint_length=9,
    normalize=True,
    preemphasis=True,
    legacy_scaling=False,
    without_samples=False,
    samples_options={},
    normalization_options={},
):
    net = rasr.FlowNetwork()

    if without_samples:
        net.add_input("samples")
        sample_input = "network:samples"
    else:
        samples_net = samples_flow(**samples_options)
        samples_mapping = net.add_net(samples_net)
        sample_input = samples_mapping[samples_net.get_output_links("samples").pop()]

    gammatone_args = {"minfreq": minfreq, "maxfreq": maxfreq, "channels": channels}
    if warp_freqbreak is not None:
        gammatone_args["warp-freqbreak"] = warp_freqbreak
    gammatone = net.add_node("signal-gammatone", "gammatone", gammatone_args)

    if preemphasis:
        node_preemphasis = net.add_node(
            "signal-preemphasis", "preemphasis", {"alpha": 1.00}
        )
        net.link(sample_input, node_preemphasis)
        net.link(node_preemphasis, gammatone)
    else:
        net.link(sample_input, gammatone)

    tempint = net.add_node(
        "signal-temporalintegration",
        "temporal-integration",
        {
            "type": tempint_type,
            "shift": tempint_shift,
            "length": tempint_length,
            "flush-before-gap": flush_before_gap,
        },
    )
    net.link(gammatone, tempint)

    if do_specint:
        specint = net.add_node(
            "signal-spectralintegration",
            "spectral-integration",
            {"type": specint_type, "shift": specint_shift, "length": specint_length},
        )
        net.link(tempint, specint)
    else:
        specint = None  # this line is here just to silence a PyCharm warning

    convert = net.add_node(
        "generic-convert-vector-vector-f32-to-vector-f32", "typeconvert"
    )
    if do_specint:
        net.link(specint, convert)
    else:
        net.link(tempint, convert)

    scaling = net.add_node(
        "generic-vector-f32-multiplication", "scaling", {"value": 0.00035}
    )
    net.link(convert, scaling)

    nonlinear = net.add_node("generic-vector-f32-power", "nonlinear", {"value": 0.1})
    net.link(scaling, nonlinear)

    cos_transform = net.add_node(
        "signal-cosine-transform", "cos_transform", {"nr-outputs": channels}
    )
    net.link(nonlinear, cos_transform)

    if normalize:
        attr = {
            "type": "mean-and-variance",
            "length": "infinity",
            "right": "infinity",
        }
        attr.update(normalization_options)
        normalization = net.add_node("signal-normalization", "gt-normalization", attr)
        net.link(cos_transform, normalization)

        if (
            legacy_scaling
        ):  # In legacy setups, features were multiplied with a scalar of 3
            post_norm_scaling = net.add_node(
                "generic-vector-f32-multiplication", "post-norm-scaling", {"value": 3}
            )
            net.link(normalization, post_norm_scaling)
            net.link(post_norm_scaling, "network:features")
        else:
            net.link(normalization, "network:features")

    else:
        net.link(cos_transform, "network:features")

    return net
