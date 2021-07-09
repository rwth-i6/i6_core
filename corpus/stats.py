__all__ = ["ExtractOovWordsFromCorpusJob", "CountCorpusWordFrequenciesJob"]

from collections import Counter
import xml.etree.cElementTree as ET

import i6_core.lib.corpus as libcorpus
from i6_core.util import uopen

from sisyphus import *

Path = setup_path(__package__)


class ExtractOovWordsFromCorpusJob(Job):
    """
    Extracts the out of vocabulary words based on a given corpus and lexicon
    """

    def __init__(self, bliss_corpus, bliss_lexicon):
        """
        :param Union[Path, str] bliss_corpus: path to corpus file
        :param Union[Path, str] bliss_lexicon: path to lexicon
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon

        self.out_oov_words = self.output_path("oov_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.bliss_lexicon, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            iv_words = {
                orth.text.upper() for orth in tree.findall(".//lemma/orth") if orth.text
            }

        with uopen(self.bliss_corpus, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            oov_words = {
                w
                for kw in tree.findall(".//recording/segment/orth")
                for w in kw.text.strip().split()
                if w.upper() not in iv_words
            }

        with uopen(self.out_oov_words, "wt") as f:
            for w in sorted(oov_words):
                f.write("%s\n" % w)


class CountCorpusWordFrequenciesJob(Job):
    """
    Extracts a list of words and their counts in the provided bliss corpus
    """

    def __init__(self, bliss_corpus):
        """
        :param Path bliss_corpus: path to corpus file
        """
        self.bliss_corpus = bliss_corpus

        self.out_word_counts = self.output_path("counts")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        c = libcorpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        words = Counter()
        for s in c.segments():
            words.update(s.orth.strip().split())

        counts = [(v, k) for k, v in words.items()]
        with uopen(self.out_word_counts, "wt") as f:
            f.write(
                "\n".join(
                    "%d\t%s" % t for t in sorted(counts, key=lambda t: (-t[0], t[1]))
                )
            )
