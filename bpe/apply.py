__all__ = ["ApplyBPEModelToLexiconJob"]

import gzip
import subprocess as sp
import os
import sys
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

from recipe.i6_asr.git import *
from recipe.i6_asr.lib.lexicon import Lexicon


class ApplyBPEModelToLexiconJob(Job):
    def __init__(self, bpe_code, lexicon_path, vocabulary_path=None):
        self.bpe_code = bpe_code
        self.lexicon_path = lexicon_path
        self.vocabulary_path = vocabulary_path

        self.subword_nmt_repo = CloneGitRepository(
            "https://github.com/rsennrich/subword-nmt.git"
        ).repository

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
