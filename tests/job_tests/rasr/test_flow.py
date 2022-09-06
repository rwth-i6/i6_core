import string

from i6_core.rasr.flow import FlowNetwork


def test_deterministic_flow_serialization():
    """
    Check if the RASR flow network is serialized in a deterministic way by
    running multiple serializations of slightly different flow networks.
    Serialization used to be non-deterministic over different python interpreter runs.
    """

    ref_net = """\
<?xml version="1.0" ?>
<network name="network">
  <node filter="a" name="{0}"/>
  <node filter="b" name="{1}"/>
  <link from="{0}" to="{1}"/>
  <node filter="c" name="{2}"/>
  <link from="{0}" to="{2}"/>
  <node filter="d" name="{3}"/>
  <link from="{1}" to="{3}"/>
  <link from="{2}" to="{3}"/>
</network>
"""

    nets = []
    for i in range(50):
        net = FlowNetwork()
        nets.append(net)
        nodes = []
        for n in "abcd":
            nodes.append(net.add_node(n, n + str(i)))
        net.link(nodes[0], nodes[1])
        net.link(nodes[0], nodes[2])
        net.link(nodes[1], nodes[3])
        net.link(nodes[2], nodes[3])

        assert str(net) == ref_net.format(*nodes), "Failed after %d iterations" % i
