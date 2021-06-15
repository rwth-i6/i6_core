__all__ = ["ApplyG2PModelJob"]

import subprocess as sp

from sisyphus import *

from recipe.i6_core.util import uopen

Path = setup_path(__package__)


class ApplyG2PModelJob(Job):
    """
    Apply a trained G2P on a word list file
    """

    def __init__(
        self,
        g2p_model,
        word_list_file,
        variants_mass=1.0,
        variants_number=1,
        g2p_path=None,
        g2p_python=None,
    ):
        """
        :param Path g2p_model:
        :param Path word_list_file: text file with a word each line
        :param float variants_mass:
        :param int variants_number:
        :param Path|str|None g2p_path:
        :param Path|str|None g2p_python:
        """

        if g2p_path is None:
            g2p_path = tk.gs.G2P_PATH
        if g2p_python is None:
            g2p_python = tk.gs.G2P_PYTHON

        self.g2p_model = g2p_model
        self.g2p_path = g2p_path
        self.g2p_python = g2p_python
        self.variants_mass = variants_mass
        self.variants_number = variants_number
        self.word_list = word_list_file

        self.out_g2p_lexicon = self.output_path("g2p.lexicon")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.out_g2p_lexicon, "wt") as out:
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
                    self.g2p_model.get_path(),
                    "-a",
                    self.word_list.get_path(),
                ],
                stdout=out,
            )
