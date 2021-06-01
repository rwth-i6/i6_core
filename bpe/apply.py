__all__ = ["ApplyBPEModelToLexiconJob", "ApplyBPEToTextJob"]

import gzip
import subprocess as sp
import os
import sys
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

from i6_core.lib.lexicon import Lexicon
from i6_core.tools.git import *


class ApplyBPEModelToLexiconJob(Job):
    def __init__(self, bpe_code, lexicon_path, vocabulary_path=None):
        self.bpe_code = bpe_code
        self.lexicon_path = lexicon_path
        self.vocabulary_path = vocabulary_path

        self.subword_nmt_repo = CloneGitRepositoryJob(
            "https://github.com/rsennrich/subword-nmt.git"
        ).out_repository

        self.converted_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon_path = tk.uncached_path(self.lexicon_path)

        lexicon = Lexicon()
        lexicon.load(lexicon_path)

        lm_tokens = set()
        for l in lexicon.lemma:
            for orth in l.orth:
                lm_tokens.add(orth)
            for synt in l.synt:
                for t in synt:
                    lm_tokens.add(t)
            for eval in l.eval:
                for t in eval:
                    lm_tokens.add(t)

        lm_tokens = list(lm_tokens)

        with open("words", "wt") as f:
            for t in lm_tokens:
                f.write("%s\n" % t)

        apply_binary = os.path.join(
            self.subword_nmt_repo.get_path(), "subword_nmt/apply_bpe.py"
        )
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words",
            "--codes",
            tk.uncached_path(self.bpe_code),
            "--output",
            "bpes",
        ]
        if self.vocabulary_path is not None:
            args += ["--vocabulary", tk.uncached_path(self.vocabulary_path)]
        sp.run(args)

        with open("bpes", "rt") as f:
            bpe_tokens = [l.strip().split() for l in f]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        for l in lexicon.lemma:
            if l.special is None and len(l.orth) > 0:
                if len(l.synt) == 0 and len(l.eval) == 0:
                    o = l.orth[0]
                    l.synt.append(w2b[o])
                    l.eval.append([o])
                if len(l.synt) > 0:
                    l.synt = [
                        sum([w2b[t] for t in token_sequence], [])
                        for token_sequence in l.synt
                    ]
                if len(l.eval) > 0:
                    l.eval = [
                        sum([w2b[t] for t in token_sequence], [])
                        for token_sequence in l.eval
                    ]

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        with gzip.open(self.converted_lexicon.get_path(), "wb") as f:
            tree.write(f, encoding="utf-8")


class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    def __init__(self, text_file, bpe_codes, bpe_vocab=None):
        """
        :param Path|str text_file: words text file to convert to bpe
        :param Path|str bpe_codes: bpe codes file
        :param Path|str|None: bpe_vocab: if provided, then merge operations that produce OOV are reverted
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab

        self.subword_nmt_repo = CloneGitRepositoryJob(
            "https://github.com/albertz/subword-nmt.git",
            branch="master",
            commit="6ba4515d684393496502b79188be13af9cad66e2",
        ).out_repository

        self.out_bpe_text = self.output_path("words_to_bpe.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        cmd = [
            sys.executable,
            os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py"),
            "--input",
            tk.uncached_path(self.text_file),
            "--codes",
            tk.uncached_path(self.bpe_codes),
            "--output",
            self.out_bpe_text.get_path(),
        ]
        if self.bpe_vocab:
            cmd += ["--vocabulary", tk.uncached_path(self.bpe_vocab)]
        sp.run(cmd)
