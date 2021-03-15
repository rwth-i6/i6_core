__all__ = ["BlissLexiconToG2PLexiconJob", "G2POutputToBlissLexicon"]

import itertools as it
import logging
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

import recipe.i6_asr.util as util


class BlissLexiconToG2PLexiconJob(Job):
    def __init__(self, bliss_lexicon):
        self.bliss_lexicon = bliss_lexicon

        self.g2p_lexicon = self.output_path("g2p.lexicon")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with util.uopen(self.bliss_lexicon, "rt") as f:
            tree = ET.parse(f)

        with open(self.g2p_lexicon.get_path(), "wt") as out:
            for lemma in tree.findall(".//lemma"):
                if lemma.get("special") is not None:
                    continue

                orth = lemma.find("orth").text.strip()
                phon = lemma.find("phon").text.strip()

                out.write("%s %s\n" % (orth, phon))


class G2POutputToBlissLexicon(Job):
    def __init__(self, iv_lexicon, g2p_output, merge=True):
        self.iv_lexicon = iv_lexicon
        self.g2p_output = g2p_output
        self.merge = merge

        self.oov_lexicon = self.output_path("oov.lexicon.gz", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with util.uopen(self.g2p_output, "rt", encoding="utf-8") as f:
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

        with util.uopen(self.iv_lexicon, "rt") as f:
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

        with util.uopen(self.oov_lexicon, "wt", encoding="utf-8") as f:
            tree = ET.ElementTree(root)
            tree.write(f, "unicode", True)
