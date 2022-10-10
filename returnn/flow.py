from sisyphus import tk
from sisyphus.delayed_ops import DelayedJoin
from typing import Optional, List, Union

from i6_core import rasr
from i6_core.returnn.training import Checkpoint


def make_precomputed_hybrid_tf_feature_flow(
    tf_graph: tk.Path,
    tf_checkpoint: Checkpoint,
    feature_tensor_name: str = "data",
    output_layer_name: str = "output",
    native_ops: Optional[Union[tk.Path, List[tk.Path]]] = None,
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Create the feature flow for a simple TF network that predicts frame-wise outputs, to be used
    in combination with the `nn-precomputed-hybrid` feature-scorer setting in RASR.

    The resulting flow is a trivial:

        <link from="<tf_fwd_input_name>" to="tf-fwd:input"/>
        <node name="tf-fwd" id="$(id)" filter="tensorflow-forward"/>
        <link from="tf-fwd:log-posteriors" to="network:features"/>

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
    :param feature_tensor_name: name of the extern data entry to feed the features to
    :param output_layer_name: the name of the output layer, it is expected that 
        "<name>/output_batch_major" exists and returns log-probs.
    :param native_ops: list of native op ".so" files to link
    :param tf_fwd_input_name: naming for the tf network input, usually no need to be changed
    :return: tensorflow-forward node flow with output link and related config
    """

    # tf flow (model scoring done in tf flow node) #
    tf_flow = rasr.FlowNetwork()
    tf_flow.add_input(tf_fwd_input_name)
    tf_flow.add_output("features")
    tf_flow.add_param("id")

    tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
    tf_flow.link(f"network:{tf_fwd_input_name}", tf_fwd + ":input")
    tf_flow.link(tf_fwd + ":log-posteriors", "network:features")

    tf_flow.config = rasr.RasrConfig()
    tf_flow.config[tf_fwd].input_map.info_0.param_name = "input"
    tf_flow.config[
        tf_fwd
    ].input_map.info_0.tensor_name = (
        f"extern_data/placeholders/{feature_tensor_name}/{feature_tensor_name}"
    )
    tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = (
        f"extern_data/placeholders/"
        f"{feature_tensor_name}/{feature_tensor_name}_dim0_size"
    )

    tf_flow.config[tf_fwd].output_map.info_0.param_name = "log-posteriors"
    tf_flow.config[
        tf_fwd
    ].output_map.info_0.tensor_name = f"{output_layer_name}/output_batch_major"

    tf_flow.config[tf_fwd].loader.type = "meta"
    tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph
    tf_flow.config[tf_fwd].loader.saved_model_file = tf_checkpoint
    if native_ops is not None:
        if isinstance(native_ops, list):
            tf_flow.config[tf_fwd].loader.required_libraries = DelayedJoin(
                native_ops, ";"
            )
        else:
            tf_flow.config[tf_fwd].loader.required_libraries = native_ops

    return tf_flow


def add_tf_flow_to_base_flow(
    base_flow: rasr.FlowNetwork,
    tf_flow: rasr.FlowNetwork,
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Integrate tf-fwd node into the regular flow network, passing the features into the input of the tf-flow net.

    :param FlowNetwork base_flow:
    :param FlowNetwork tf_flow:
    :param str tf_fwd_input_name: see: get_tf_flow()
    :rtype: Combined FlowNetwork
    """
    assert (
        len(base_flow.outputs) == 1
    ), "Not implemented otherwise"  # see hard coded tf-fwd input
    base_output = list(base_flow.outputs)[0]

    input_name = tf_fwd_input_name

    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    tf_mapping = feature_flow.add_net(tf_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(
        base_flow, base_mapping, tf_flow, tf_mapping, {base_output: input_name}
    )
    feature_flow.interconnect_outputs(tf_flow, tf_mapping)

    # ensure cache_mode as base feature net
    feature_flow.add_flags(base_flow.flags)

    return feature_flow
