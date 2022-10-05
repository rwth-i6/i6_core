__all__ = ["ApplySPMModelToLexiconJob"]

import logging
import subprocess as sp
import os
import sys
import xml.etree.ElementTree as ET

from sisyphus import *

from i6_core.lib.lexicon import Lexicon
import i6_core.util as util


try:
    import sentencepiece
except ImportError:
    if (
        not hasattr(gs, "WARNING_NO_SENTENCEPIECE")
        or gs.WARNING_NO_SENTENCEPIECE is True
    ):
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make sure it is installed "
            "in the python environment running the Sisyphus worker. To suppress this warning set "
            "'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class ApplySPMModelToLexiconJob(Job):
    """
    Apply sentencepiece model to a Bliss lexicon file
    """

    def __init__(self, bliss_lexicon, spm_model):
        """
        :param Path bliss_lexicon:
        :param Path spm_model:
        :param Path|str|None sentencepiece_repo:
        """
        self.bliss_lexicon = bliss_lexicon
        self.spm_model = spm_model

        self.out_converted_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon_path = self.bliss_lexicon.get_path()

        lexicon = Lexicon()
        lexicon.load(lexicon_path)

        lm_tokens = set()
        for l in lexicon.lemmata:
            for orth in l.orth:
                lm_tokens.add(orth)
            for token in l.synt or []:  # l.synt can be None
                lm_tokens.add(token)
            for eval in l.eval:
                for t in eval:
                    lm_tokens.add(t)

        lm_tokens = list(lm_tokens)
        spm = sentencepiece.SentencePieceProcessor(model_file=self.spm_model.get_path())
        bpe_tokens = []
        for t in lm_tokens:
            bpe_tokens.append(spm.encode(t, out_type=str))

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        for l in lexicon.lemmata:
            if l.special is None and len(l.orth) > 0:
                if not l.synt and len(l.eval) == 0:
                    o = l.orth[0]
                    l.synt = w2b[o]
                    l.eval.append([o])
                if l.synt:
                    new_synt = []
                    for token in l.synt:
                        new_synt += w2b[token] if token in w2b else [token]
                    l.synt = new_synt
                if len(l.eval) > 0:
                    l.eval = [
                        sum([w2b[t] for t in token_sequence], [])
                        for token_sequence in l.eval
                    ]

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        util.write_xml(self.out_converted_lexicon.get_path(), tree)
