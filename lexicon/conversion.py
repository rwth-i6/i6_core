import collections
import gzip
import os.path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from sisyphus import *

Path = setup_path(__package__)

import i6_core.lib.lexicon as lexicon
from i6_core.util import uopen


class LexiconToWordListJob(Job):
    def __init__(self, bliss_lexicon, apply_filter=True):
        self.set_vis_name("Lexicon to Word List")

        self.bliss_lexicon = bliss_lexicon
        self.apply_filter = apply_filter

        self.out_word_list = self.output_path("words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.bliss_lexicon), "r") as lexicon_file:
            lexicon = ET.fromstring(lexicon_file.read())
            words = set()
            for e in lexicon.findall("./lemma/orth"):
                if (
                    e.text is not None
                    and len(e.text) > 0
                    and not (e.text.startswith("[") and self.apply_filter)
                ):
                    words.add(e.text)

        with open(self.out_word_list.get_path(), "w") as word_file:
            for w in sorted(words):
                word_file.write("%s\n" % w)


class FilterLexiconByWordListJob(Job):
    def __init__(self, bliss_lexicon, word_list, case_sensitive=False):
        self.set_vis_name("Filter Lexicon by Word List")

        self.bliss_lexicon = bliss_lexicon
        self.word_list = word_list
        self.case_sensitive = case_sensitive

        self.out_bliss_lexicon = self.output_path(
            os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        transform = (lambda s: s.lower()) if self.case_sensitive else (lambda s: s)

        with uopen(tk.uncached_path(self.bliss_lexicon), "r") as lexicon_file:
            old_lexicon = ET.fromstring(lexicon_file.read())

        with uopen(tk.uncached_path(self.word_list), "r") as words_file:
            words = set([transform(w.strip()) for w in words_file.readlines()])

        root = ET.Element("lexicon")
        root.append(old_lexicon.find("phoneme-inventory"))
        for lemma in old_lexicon.findall("lemma"):
            if any(
                transform(orth.text) in words
                or (orth.text is not None and orth.text.startswith("["))
                for orth in lemma.findall("orth")
            ):
                root.append(lemma)

        with uopen(self.out_bliss_lexicon.get_path(), "w") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(root, "unicode"))


class LexiconUniqueOrthJob(Job):
    def __init__(self, bliss_lexicon):
        self.set_vis_name("Make Lexicon Orths Unique")

        self.bliss_lexicon = bliss_lexicon

        self.out_bliss_lexicon = self.output_path(
            os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.bliss_lexicon), "r") as lexicon_file:
            old_lexicon = ET.fromstring(lexicon_file.read())

        root = ET.Element("lexicon")
        root.append(old_lexicon.find("phoneme-inventory"))

        lemmas = collections.OrderedDict()
        for lemma in old_lexicon.findall("lemma"):
            if "special" in lemma.attrib:
                root.append(lemma)
            elif len(lemma.findall("orth")) != 1:
                root.append(lemma)
            else:
                orth = lemma.find("orth").text

                if orth not in lemmas:
                    lemmas[orth] = {"phon": set(), "synt": set(), "eval": set()}

                lemmas[orth]["phon"].update(e.text for e in lemma.findall("phon"))
                lemmas[orth]["synt"].update(
                    lemma.findall("synt")
                )  # as synt can contain sub elements
                lemmas[orth]["eval"].update(e.text for e in lemma.findall("eval"))

        for orth, lemma in lemmas.items():
            el = ET.SubElement(root, "lemma")

            o = ET.SubElement(el, "orth")
            o.text = orth

            for phon in lemma["phon"]:
                p = ET.SubElement(el, "phon")
                p.text = phon

            for synt in lemma["synt"]:
                el.append(synt)

            for eval in lemma["eval"]:
                ev = ET.SubElement(el, "eval")
                ev.text = eval

        with uopen(self.out_bliss_lexicon.get_path(), "w") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(root, "unicode"))


class GraphemicLexiconFromWordListJob(Job):
    default_transforms = {".": "DOT", "+": "PLUS", "{": "LBR", "}": "RBR"}

    def __init__(
        self, word_list_file, add_unknown=False, add_noise=False, transforms=None
    ):
        self.add_unknown = add_unknown
        self.add_noise = add_noise
        self.transforms = (
            transforms if transforms is not None else self.default_transforms
        )
        self.word_list_file = word_list_file

        self.out_bliss_lexicon = self.output_path("grapheme.lexicon", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.word_list_file), "rt") as f:
            words = [l.strip() for l in f]

        phonemes = set()
        for w in words:
            phonemes.update(w)
        phonemes.discard(" ")  # just in case

        lex = lexicon.Lexicon()
        lex.add_phoneme("sil", variation="none")
        for p in sorted(phonemes):
            p = self.transforms.get(p, p)
            lex.add_phoneme(p, "context")
        if self.add_unknown:
            lex.add_phoneme("unk", "none")
        if self.add_noise:
            lex.add_phoneme("noise", "none")

        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["sil"],
                synt=[""],
                special="silence",
                eval=[""],
            )
        )
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SENTENCE_BEGIN]"], synt=["<s>"], special="sentence-begin"
            )
        )
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SENTENCE_END]"], synt=["</s>"], special="sentence-end"
            )
        )
        if self.add_unknown:
            lex.add_lemma(
                lexicon.Lemma(orth=["[UNKNOWN]"], phon=["unk"], special="unknown")
            )
        if self.add_noise:
            lex.add_lemma(
                lexicon.Lemma(
                    orth=["[NOISE]"], phon=["noise"], synt=[""], special="unknown"
                )
            )

        for w in words:
            l = lexicon.Lemma()
            l.orth.append(w)
            l.phon.append(" " + " ".join(self.transforms.get(p, p) for p in w) + " ")
            lex.add_lemma(l)

        with uopen(self.out_bliss_lexicon.get_path(), "w") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(lex.to_xml(), "unicode"))
