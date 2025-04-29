"""
Library for the RASR Lexicon files

For format details visit: `https://www-i6.informatik.rwth-aachen.de/rwth-asr/manual/index.php/Lexicon`_
"""
from __future__ import annotations

__all__ = ["Lemma", "Lexicon"]

from collections import OrderedDict
import itertools
from typing import Optional, List, Set
import xml.etree.ElementTree as ET

from i6_core.util import uopen


class Lemma:
    """
    Represents a lemma of a lexicon
    """

    def __init__(
        self,
        orth: Optional[List[str]] = None,
        phon: Optional[List[str]] = None,
        synt: Optional[List[str]] = None,
        eval: Optional[List[List[str]]] = None,
        special: Optional[str] = None,
    ):
        """
        :param orth: list of spellings used in the training data
        :param phon: list of pronunciation variants. Each str should
            contain a space separated string of phonemes from the phoneme-inventory.
        :param synt: list of LM tokens that form a single token sequence.
            This sequence is used as the language model representation.
        :param eval: list of output representations. Each
            sublist should contain one possible transcription (token sequence) of this lemma
            that is scored against the reference transcription.
        :param special: assigns special property to a lemma.
            Supported values: "silence", "unknown", "sentence-boundary",
            or "sentence-begin" / "sentence-end"
        """
        self.orth = [] if orth is None else orth
        self.phon = [] if phon is None else phon
        self.synt = synt
        self.eval = [] if eval is None else eval
        self.special = special
        if isinstance(synt, list):
            assert not (len(synt) > 0 and isinstance(synt[0], list)), (
                "providing list of list is no longer supported for the 'synt' parameter "
                "and can be safely changed into a single list"
            )

    def to_xml(self):
        """
        :return: xml representation
        :rtype:  ET.Element
        """
        attrib = {"special": self.special} if self.special is not None else {}
        res = ET.Element("lemma", attrib=attrib)
        for o in self.orth:
            el = ET.SubElement(res, "orth")
            el.text = o
        for p in self.phon:
            el = ET.SubElement(res, "phon")
            el.text = p
        if self.synt is not None:
            el = ET.SubElement(res, "synt")
            for token in self.synt:
                el2 = ET.SubElement(el, "tok")
                el2.text = token
        for e in self.eval:
            el = ET.SubElement(res, "eval")
            for t in e:
                el2 = ET.SubElement(el, "tok")
                el2.text = t
        return res

    @classmethod
    def from_element(cls, e):
        """
        :param ET.Element e:
        :rtype: Lemma
        """
        orth = []
        phon = []
        synt = []
        eval = []
        special = None
        if "special" in e.attrib:
            special = e.attrib["special"]
        for orth_element in e.findall(".//orth"):
            orth.append(orth_element.text.strip() if orth_element.text is not None else "")
        for phon_element in e.findall(".//phon"):
            phon.append(phon_element.text.strip() if phon_element.text is not None else "")
        for synt_element in e.findall(".//synt"):
            tokens = []
            for token_element in synt_element.findall(".//tok"):
                tokens.append(token_element.text.strip() if token_element.text is not None else "")
            synt.append(tokens)
        for eval_element in e.findall(".//eval"):
            tokens = []
            for token_element in eval_element.findall(".//tok"):
                tokens.append(token_element.text.strip() if token_element.text is not None else "")
            eval.append(tokens)
        synt = None if not synt else synt[0]
        return Lemma(orth, phon, synt, eval, special)

    def _equals(self, other: Lemma, *, same_order: bool = True) -> bool:
        """
        Check for lemma equality.

        :param other: Other lemma to compare :param:`self` to.
        :param same_order: Whether the order in the different lemma elements matters or not.
        :return: Whether :param:`self` and :param:`other` are equal or not.
        """
        if same_order:
            return (
                self.orth == other.orth
                and self.phon == other.phon
                and self.special == other.special
                and self.synt == other.synt
                and self.eval == other.eval
            )
        else:
            if self.synt is not None and other.synt is not None:
                equal_synt = set(self.synt) == set(other.synt)
            else:
                equal_synt = self.synt == other.synt

            return (
                set(self.orth) == set(other.orth)
                and set(self.phon) == set(other.phon)
                and self.special == other.special
                and equal_synt
                and set(itertools.chain(*self.eval)) == set(itertools.chain(*other.eval))
            )

    def __eq__(self, other: Lemma) -> bool:
        return self._equals(other, same_order=False)

    def __ne__(self, other: Lemma) -> bool:
        return not self.__eq__(other)


class Lexicon:
    """
    Represents a bliss lexicon, can be read from and written to .xml files
    """

    def __init__(self):
        self.phonemes = OrderedDict()  # type: OrderedDict[str, str] # symbol => variation
        self.lemmata = []  # type: List[Lemma]

    def add_phoneme(self, symbol, variation="context"):
        """
        :param str symbol: representation of one phoneme
        :param str variation: possible values: "context" or "none".
            Use none for context independent phonemes like silence and noise.
        """
        self.phonemes[symbol] = variation

    def remove_phoneme(self, symbol):
        """
        :param str symbol:
        """
        del self.phonemes[symbol]

    def add_lemma(self, lemma):
        """
        :param Lemma lemma:
        """
        assert isinstance(lemma, Lemma)
        self.lemmata.append(lemma)

    def load(self, path):
        """
        :param str path: bliss lexicon .xml or .xml.gz file
        """
        with uopen(path, "rt") as f:
            root = ET.parse(f)

        for phoneme in root.findall(".//phoneme-inventory/phoneme"):
            symbol = phoneme.find(".//symbol").text.strip()
            variation_element = phoneme.find(".//variation")
            variation = "context"
            if variation_element is not None:
                variation = variation_element.text.strip()
            self.add_phoneme(symbol, variation)

        for lemma in root.findall(".//lemma"):
            l = Lemma.from_element(lemma)
            self.add_lemma(l)

    def to_xml(self):
        """
        :return: xml representation, can be used with `util.write_xml`
        :rtype: ET.Element
        """
        root = ET.Element("lexicon")

        pi = ET.SubElement(root, "phoneme-inventory")
        for symbol, variation in self.phonemes.items():
            p = ET.SubElement(pi, "phoneme")
            s = ET.SubElement(p, "symbol")
            s.text = symbol
            v = ET.SubElement(p, "variation")
            v.text = variation

        for l in self.lemmata:
            root.append(l.to_xml())

        return root
