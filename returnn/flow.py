from sisyphus import tk
from sisyphus.delayed_ops import DelayedJoin
from typing import Optional, List, Union, Dict

from i6_core import rasr
from i6_core.returnn.training import Checkpoint


def make_precomputed_hybrid_feature_flow(
    backend: str,
    rasr_config: rasr.RasrConfig,
    fwd_input_name: str = "fwd-input",
) -> rasr.FlowNetwork:
    """
    Create the feature flow for a simple TF/ONNX network that predicts frame-wise outputs, to be used
    in combination with the `nn-precomputed-hybrid` feature-scorer setting in RASR.

    The resulting flow is a trivial (for ONNX, the "tf" is replaced by "onnx"):

        <link from="<tf_fwd_input_name>" to="tf-fwd:input"/>
        <node name="tf-fwd" id="$(id)" filter="tensorflow-forward"/>
        <link from="tf-fwd:log-posteriors" to="network:features"/>

    :param backend: "tf" or "onnx"
    :param rasr_config: rasr config for the forward node
    :param fwd_input_name: naming for the tf network input, usually no need to be changed
    :return: tensorflow-/onnx-forward node flow with output link and related config
    """

    # flow (model scoring done in tf/onnx flow node)
    flow = rasr.FlowNetwork()
    flow.add_input(fwd_input_name)
    flow.add_output("features")
    flow.add_param("id")

    node_filter = {"tf": "tensorflow-forward", "onnx": "onnx-forward"}[backend]
    fwd_node = flow.add_node(node_filter, f"{backend}-fwd", {"id": "$(id)"})
    flow.link(f"network:{fwd_input_name}", fwd_node + ":input")
    flow.link(fwd_node + ":log-posteriors", "network:features")

    flow.config = rasr.RasrConfig()
    flow.config[fwd_node] = rasr_config

    return flow


def make_precomputed_hybrid_tf_feature_flow(
    tf_graph: tk.Path,
    tf_checkpoint: Checkpoint,
    extern_data_name: str = "data",
    output_layer_name: str = "output",
    native_ops: Optional[Union[tk.Path, List[tk.Path]]] = None,
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Create the feature flow for a simple TF network that predicts frame-wise outputs,
    see make_precomputed_hybrid_feature_flow.

    With the config settings:

        [flf-lattice-tool.network.recognizer.feature-extraction.tf-fwd.input-map.info-0]
        param-name             = input
        seq-length-tensor-name = extern_data/placeholders/<feature_tensor_name>/<feature_tensor_name>_dim0_size
        tensor-name            = extern_data/placeholders/<feature_tensor_name>/<feature_tensor_name>

        [flf-lattice-tool.network.recognizer.feature-extraction.tf-fwd.loader]
        meta-graph-file    = <tf_graph>
        required-libraries = <native_ops>
        saved-model-file   = <tf_checkpoint>
        type               = meta

        [flf-lattice-tool.network.recognizer.feature-extraction.tf-fwd.output-map.info-0]
        param-name  = <output_type>
        tensor-name = <output_tensor_name>/output_batch_major

    :param tf_graph: usually the output of a CompileTFGraphJob
    :param tf_checkpoint: the checkpoint to load the model from, e.g. from a ReturnnTrainingJob or similar
    :param extern_data_name: name of the extern data entry to feed the features to
    :param output_layer_name: the name of the output layer, it is expected that
        "<name>/output_batch_major" exists and returns log-probs.
    :param native_ops: list of native op ".so" files to link
    :param tf_fwd_input_name: naming for the tf network input, usually no need to be changed
    :return: tensorflow-forward node flow with output link and related config
    """

    rasr_config = rasr.RasrConfig()
    rasr_config.input_map.info_0.param_name = "input"
    rasr_config.input_map.info_0.tensor_name = f"extern_data/placeholders/{extern_data_name}/{extern_data_name}"
    rasr_config.input_map.info_0.seq_length_tensor_name = (
        f"extern_data/placeholders/" f"{extern_data_name}/{extern_data_name}_dim0_size"
    )

    rasr_config.output_map.info_0.param_name = "log-posteriors"
    rasr_config.output_map.info_0.tensor_name = f"{output_layer_name}/output_batch_major"

    rasr_config.loader.type = "meta"
    rasr_config.loader.meta_graph_file = tf_graph
    rasr_config.loader.saved_model_file = tf_checkpoint
    if native_ops is not None:
        if isinstance(native_ops, list):
            rasr_config.loader.required_libraries = DelayedJoin(native_ops, ";")
        else:
            rasr_config.loader.required_libraries = native_ops
    return make_precomputed_hybrid_feature_flow(
        backend="tf",
        rasr_config=rasr_config,
        fwd_input_name=tf_fwd_input_name,
    )


def make_precomputed_hybrid_onnx_feature_flow(
    onnx_model: tk.Path,
    io_map: Dict[str, str],
    onnx_fwd_input_name: str = "onnx-fwd-input",
    cpu: int = 1,
) -> rasr.FlowNetwork:
    """
    Create the feature flow for a simple ONNX network that predicts frame-wise outputs,
    see make_precomputed_hybrid_feature_flow.

    With the config settings:

        [flf-lattice-tool.network.recognizer.feature-extraction.onnx-fwd.io-map]
        features      = data
        features-size = data_len
        output        = classes

        [flf-lattice-tool.network.recognizer.feature-extraction.onnx-fwd.session]
        file                 = <onnx_file>
        inter-op-num-threads = 2
        intra-op-num-threads = 2

    :param onnx_model: usually the output of a OnnxExportJob
    :param io_map: e.g. {"features": "data", "output": "classes"}
    :param onnx_fwd_input_name: naming for the onnx network input, usually no need to be changed
    :param cpu: number of CPUs to use
    :return: onnx-forward node flow with output link and related config
    """

    rasr_config = rasr.RasrConfig()
    for k, v in io_map.items():
        rasr_config.io_map[k] = v

    rasr_config.session.file = onnx_model
    rasr_config.session.inter_op_num_threads = cpu
    rasr_config.session.intra_op_num_threads = cpu

    return make_precomputed_hybrid_feature_flow(
        backend="onnx",
        rasr_config=rasr_config,
        fwd_input_name=onnx_fwd_input_name,
    )


def add_fwd_flow_to_base_flow(
    base_flow: rasr.FlowNetwork,
    fwd_flow: rasr.FlowNetwork,
    fwd_input_name: str = "fwd-input",
) -> rasr.FlowNetwork:
    """
    Integrate tf- or onnx-fwd node into a regular flow network, passing the features to the input of the forwarding net.

    :param FlowNetwork base_flow:
    :param FlowNetwork fwd_flow:
    :param str fwd_input_name: see: make_precomputed_hybrid_feature_flow()
    :rtype: Combined FlowNetwork
    """
    assert len(base_flow.outputs) == 1, "Not implemented otherwise"  # see hard coded fwd input
    base_output = list(base_flow.outputs)[0]

    input_name = fwd_input_name

    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    fwd_mapping = feature_flow.add_net(fwd_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(base_flow, base_mapping, fwd_flow, fwd_mapping, {base_output: input_name})
    feature_flow.interconnect_outputs(fwd_flow, fwd_mapping)

    # ensure cache_mode as base feature net
    feature_flow.add_flags(base_flow.flags)

    return feature_flow


def add_tf_flow_to_base_flow(
    base_flow: rasr.FlowNetwork,
    tf_flow: rasr.FlowNetwork,
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Keep old name to avoid breaking setups
    """
    return add_fwd_flow_to_base_flow(base_flow, tf_flow, tf_fwd_input_name)
