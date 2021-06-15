__all__ = ["BlissLexiconToG2PLexiconJob", "G2POutputToBlissLexiconJob"]

import itertools as it
import logging
import xml.etree.ElementTree as ET

from sisyphus import *

from i6_core.util import uopen

Path = setup_path(__package__)


class BlissLexiconToG2PLexiconJob(Job):
    """
    Convert a bliss lexicon into a g2p compatible lexicon for training
    """

    def __init__(self, bliss_lexicon):
        """
        :param Path bliss_lexicon:
        """
        self.bliss_lexicon = bliss_lexicon

        self.out_g2p_lexicon = self.output_path("g2p.lexicon")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.bliss_lexicon, "rt") as f:
            tree = ET.parse(f)
        with uopen(self.out_g2p_lexicon, "wt") as out:
            for lemma in tree.findall(".//lemma"):
                if lemma.get("special") is not None:
                    continue

                orth = lemma.find("orth").text.strip()
                phon = lemma.find("phon").text.strip()

                out.write("%s %s\n" % (orth, phon))


class G2POutputToBlissLexiconJob(Job):
    """
    Convert a g2p applied word list file (g2p lexicon) into a bliss lexicon
    """

    def __init__(self, iv_bliss_lexicon, g2p_lexicon, merge=True):
        """
        :param Path iv_bliss_lexicon: bliss lexicon as reference for the phoneme inventory
        :param Path g2p_lexicon: from ApplyG2PModelJob.out_g2p_lexicon
        :param bool merge: merge the g2p lexicon into the iv_bliss_lexicon instead of
            only taking the phoneme inventory
        """
        self.iv_bliss_lexicon = iv_bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.merge = merge

        self.out_oov_lexicon = self.output_path("oov.lexicon.gz", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.g2p_lexicon, "rt", encoding="utf-8") as f:
            oov_words = dict()
            for orth, data in it.groupby(
                map(lambda line: line.strip().split("\t"), f), lambda t: t[0]
            ):
                oov_words[orth] = []
                for d in data:
                    if len(d) == 4:
                        oov_words[orth].append(d[3])
                    elif len(d) < 4:
                        logging.warning(
                            'No pronunciation found for orthography "{}"'.format(orth)
                        )
                    else:
                        logging.warning(
                            'Did not fully parse entry for orthography "{}"'.format(
                                orth
                            )
                        )

        with uopen(self.iv_bliss_lexicon, "rt") as f:
            iv_lexicon = ET.parse(f)

        if self.merge:
            root = iv_lexicon.getroot()
        else:
            root = ET.Element("lexicon")
            root.append(iv_lexicon.find("phoneme-inventory"))

        for orth, prons in oov_words.items():
            lemma = ET.SubElement(root, "lemma")
            ET.SubElement(lemma, "orth").text = orth
            for pron in prons:
                ET.SubElement(lemma, "phon").text = pron

        with uopen(self.out_oov_lexicon, "wt", encoding="utf-8") as f:
            tree = ET.ElementTree(root)
            tree.write(f, "unicode", True)
