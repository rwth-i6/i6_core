__all__ = ["ApplyG2PModelJob"]

import subprocess as sp

from sisyphus import *

Path = setup_path(__package__)


class ApplyG2PModelJob(Job):
    def __init__(
        self,
        g2p_model,
        word_list,
        variants_mass=1.0,
        variants_number=1,
        g2p_path=None,
        g2p_python=None,
    ):
        if g2p_path is None:
            g2p_path = tk.gs.G2P_PATH
        if g2p_python is None:
            g2p_python = tk.gs.G2P_PYTHON

        self.g2p_model = g2p_model
        self.g2p_path = g2p_path
        self.g2p_python = g2p_python
        self.variants_mass = variants_mass
        self.variants_number = variants_number
        self.word_list = word_list

        self.g2p_lexicon = self.output_path("g2p.lexicon")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.g2p_lexicon.get_path(), "wt") as out:
            sp.check_call(
                [
                    self.g2p_python,
                    self.g2p_path,
                    "-e",
                    "utf-8",
                    "-V",
                    str(self.variants_mass),
                    "--variants-number",
                    str(self.variants_number),
                    "-m",
                    tk.uncached_path(self.g2p_model),
                    "-a",
                    tk.uncached_path(self.word_list),
                ],
                stdout=out,
            )
