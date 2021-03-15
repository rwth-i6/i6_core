__all__ = ["ExtractOovWordsFromCorpus"]

import xml.etree.cElementTree as ET

from recipe.i6_asr.util import uopen

from sisyphus import *

Path = setup_path(__package__)


class ExtractOovWordsFromCorpus(Job):
    """
    Extracts the out of vocabulary words based on a given corpus and lexicon
    """

    def __init__(self, corpus, lexicon):
        """
        :param Union[Path, str] corpus: path to corpus file
        :param Union[Path, str] lexicon: path to lexicon
        """
        self.corpus = corpus
        self.lexicon = lexicon

        self.oov_words = self.output_path("oov_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.lexicon, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            iv_words = {
                orth.text.upper() for orth in tree.findall(".//lemma/orth") if orth.text
            }

        with uopen(self.corpus, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            oov_words = {
                w
                for kw in tree.findall(".//recording/segment/orth")
                for w in kw.text.strip().split()
                if w.upper() not in iv_words
            }

        with uopen(self.oov_words, "wt") as f:
            for w in sorted(oov_words):
                f.write("%s\n" % w)
