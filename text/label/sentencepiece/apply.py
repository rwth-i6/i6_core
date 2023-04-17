__all__ = ["ApplySPMModelToLexiconJob"]

import logging

from sisyphus import *

from i6_core.lib.lexicon import Lexicon
from i6_core.text.label.subword_utils import (
    get_lm_tokens_from_lexicon,
    word_to_subword_in_lexicon,
)
from i6_core import util

try:
    import sentencepiece
except ImportError:
    if (
        not hasattr(gs, "WARNING_NO_SENTENCEPIECE")
        or gs.WARNING_NO_SENTENCEPIECE is True
    ):
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make "
            "sure it is installed in the python environment running the Sisyphus worker. To "
            "suppress this warning set 'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class ApplySPMModelToLexiconJob(Job):
    """
    Apply sentencepiece model to a Bliss lexicon file
    """

    def __init__(self, bliss_lexicon, spm_model):
        """
        :param Path bliss_lexicon:
        :param Path spm_model:
        """
        self.bliss_lexicon = bliss_lexicon
        self.spm_model = spm_model

        self.out_converted_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())

        lm_tokens = get_lm_tokens_from_lexicon(lexicon)
        spm = sentencepiece.SentencePieceProcessor(model_file=self.spm_model.get_path())
        spm_tokens = []
        for t in lm_tokens:
            spm_tokens.append(spm.encode(t, out_type=str))

        tree = word_to_subword_in_lexicon(lexicon, lm_tokens, spm_tokens)
        util.write_xml(self.out_converted_lexicon.get_path(), tree)
