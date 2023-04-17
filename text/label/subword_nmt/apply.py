__all__ = ["ApplyBPEModelToLexiconJob", "ApplyBPEToTextJob"]

import subprocess as sp
import os
import sys

from sisyphus import *

Path = setup_path(__package__)

from i6_core.lib.lexicon import Lexicon
from i6_core.text.label.subword_utils import (
    get_lm_tokens_from_lexicon,
    word_to_subword_in_lexicon,
)
import i6_core.util as util


class ApplyBPEModelToLexiconJob(Job):
    """
    Apply BPE codes to a Bliss lexicon file
    """

    def __init__(self, bliss_lexicon, bpe_codes, bpe_vocab=None, subword_nmt_repo=None):
        """
        :param Path bliss_lexicon:
        :param Path bpe_codes:
        :param Path|None bpe_vocab:
        :param Path|str|None subword_nmt_repo:
        """
        self.bliss_lexicon = bliss_lexicon
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = (
            subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        )

        self.out_converted_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())
        lm_tokens = get_lm_tokens_from_lexicon(lexicon)

        with util.uopen("words", "wt") as f:
            for t in lm_tokens:
                f.write("%s\n" % t)

        apply_binary = os.path.join(
            tk.uncached_path(self.subword_nmt_repo), "subword_nmt/apply_bpe.py"
        )
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words",
            "--codes",
            self.bpe_codes.get_path(),
            "--output",
            "bpes",
        ]
        if self.bpe_vocab is not None:
            args += ["--vocabulary", self.bpe_vocab.get_path()]
        sp.run(args, check=True)

        with util.uopen("bpes", "rt") as f:
            bpe_tokens = [l.strip().split() for l in f]

        tree = word_to_subword_in_lexicon(lexicon, lm_tokens, bpe_tokens)
        with util.uopen(self.out_converted_lexicon.get_path(), "wb") as f:
            tree.write(f, encoding="utf-8")


class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    def __init__(self, text_file, bpe_codes, bpe_vocab=None, subword_nmt_repo=None):
        """
        :param Path text_file: words text file to convert to bpe
        :param Path bpe_codes: bpe codes file
        :param Path|None bpe_vocab: if provided, then merge operations that produce OOV are reverted
        :param Path|str|None subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = (
            subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        )

        self.out_bpe_text = self.output_path("words_to_bpe.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        cmd = [
            sys.executable,
            os.path.join(tk.uncached_path(self.subword_nmt_repo), "apply_bpe.py"),
            "--input",
            self.text_file.get_path(),
            "--codes",
            self.bpe_codes.get_path(),
            "--output",
            self.out_bpe_text.get_path(),
        ]

        if self.bpe_vocab:
            cmd += ["--vocabulary", self.bpe_vocab.get_path()]

        util.create_executable("apply_bpe.sh", cmd)
        sp.run(cmd, check=True)
