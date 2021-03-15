__all__ = ["Lemma", "Lexicon"]

from collections import OrderedDict
import gzip
import xml.etree.ElementTree as ET


class Lemma:
    def __init__(self, orth=None, phon=None, synt=None, eval=None, special=None):
        self.orth = [] if orth is None else orth
        self.phon = [] if phon is None else phon
        self.synt = [] if synt is None else synt
        self.eval = [] if eval is None else eval
        self.special = special

    def to_xml(self):
        attrib = {"special": self.special} if self.special is not None else {}
        res = ET.Element("lemma", attrib=attrib)
        for o in self.orth:
            el = ET.SubElement(res, "orth")
            el.text = o
        for p in self.phon:
            el = ET.SubElement(res, "phon")
            el.text = p
        for s in self.synt:
            el = ET.SubElement(res, "synt")
            for t in s:
                el2 = ET.SubElement(el, "tok")
                el2.text = t
        for e in self.eval:
            el = ET.SubElement(res, "eval")
            for t in e:
                el2 = ET.SubElement(el, "tok")
                el2.text = t

        return res

    @classmethod
    def from_element(cls, e):
        orth = []
        phon = []
        synt = []
        eval = []
        special = None
        if "special" in e.attrib:
            special = e.attrib["special"]
        for orth_element in e.findall(".//orth"):
            orth.append(
                orth_element.text.strip() if orth_element.text is not None else ""
            )
        for phon_element in e.findall(".//phon"):
            phon.append(
                phon_element.text.strip() if phon_element.text is not None else ""
            )
        for synt_element in e.findall(".//synt"):
            tokens = []
            for token_element in synt_element.findall(".//tok"):
                tokens.append(
                    token_element.text.strip() if token_element.text is not None else ""
                )
            synt.append(tokens)
        for eval_element in e.findall(".//eval"):
            tokens = []
            for token_element in eval_element.findall(".//tok"):
                tokens.append(
                    token_element.text.strip() if token_element.text is not None else ""
                )
            eval.append(tokens)
        return Lemma(orth, phon, synt, eval, special)


class Lexicon:
    def __init__(self):
        self.phonemes = OrderedDict()  # symbol => variation
        self.lemma = []

    def add_phoneme(self, symbol, variation="context"):
        self.phonemes[symbol] = variation

    def remove_phoneme(self, symbol):
        del self.phonemes[symbol]

    def add_lemma(self, lemma):
        assert isinstance(lemma, Lemma)
        self.lemma.append(lemma)

    def load(self, path):
        open_fun = gzip.open if path.endswith(".gz") else open

        with open_fun(path, "rt") as f:
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
        root = ET.Element("lexicon")

        pi = ET.SubElement(root, "phoneme-inventory")
        for symbol, variation in self.phonemes.items():
            p = ET.SubElement(pi, "phoneme")
            s = ET.SubElement(p, "symbol")
            s.text = symbol
            v = ET.SubElement(p, "variation")
            v.text = variation

        for l in self.lemma:
            root.append(l.to_xml())

        return root
