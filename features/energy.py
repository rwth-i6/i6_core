__all__ = ["EnergyJob", "energy_flow"]

import copy

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def EnergyJob(crp, energy_options=None, extra_config=None, extra_post_config=None):
    if energy_options is None:
        energy_options = {}
    else:
        energy_options = copy.deepcopy(energy_options)
    if "samples_options" not in energy_options:
        energy_options["samples_options"] = {}
    energy_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = energy_flow(**energy_options)

    port_name_mapping = {"energy": "energy"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="Energy",
        rtf=0.1,
        mem=2,
        extra_config=extra_config,
        extra_post_config=extra_post_config,
    )


def energy_flow(without_samples=False, samples_options={}, fft_options={}):
    net = rasr.FlowNetwork()

    if without_samples:
        net.add_input("samples")
        fft_net = fft_flow(**fft_options)
        fft_mapping = net.add_net(fft_net)
        net.interconnect_inputs(fft_net, fft_mapping)
    else:
        samples_net = samples_flow(**samples_options)
        samples_mapping = net.add_net(samples_net)
        fft_net = fft_flow(**fft_options)
        fft_mapping = net.add_net(fft_net)
        net.interconnect(samples_net, samples_mapping, fft_net, fft_mapping)

    energy = net.add_node("generic-vector-f32-norm", "energy", {"value": 1})
    net.link(fft_mapping[fft_net.get_output_links("amplitude-spectrum").pop()], energy)

    convert_energy_to_vector = net.add_node(
        "generic-convert-f32-to-vector-f32", "convert-energy-to-vector"
    )
    net.link(energy, convert_energy_to_vector)

    energy_normalization = net.add_node(
        "signal-normalization",
        "energy-normalization",
        {"type": "divide-by-mean", "length": "infinite", "right": "infinite"},
    )
    net.link(convert_energy_to_vector, energy_normalization)

    convert_energy_to_scalar = net.add_node(
        "generic-convert-vector-f32-to-f32", "convert-energy-vector-to-scalar"
    )
    net.link(energy_normalization, convert_energy_to_scalar)
    net.link(convert_energy_to_scalar, "network:energy")

    return net
