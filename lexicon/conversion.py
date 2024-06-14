import collections
import copy
import json
import logging
import os.path
import re
from typing import List, Optional, Tuple, Union
import xml.dom.minidom
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

import i6_core.lib.lexicon as lexicon
from i6_core.util import uopen, write_xml


class LexiconToWordListJob(Job):
    def __init__(self, bliss_lexicon: Path, apply_filter: bool = True):
        self.set_vis_name("Lexicon to Word List")

        self.bliss_lexicon = bliss_lexicon
        self.apply_filter = apply_filter

        self.out_word_list = self.output_path("words", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.bliss_lexicon.get_path(), "r") as lexicon_file:
            lexicon = ET.fromstring(lexicon_file.read())
            words = set()
            for e in lexicon.findall("./lemma/orth"):
                if e.text is not None and len(e.text) > 0 and not (e.text.startswith("[") and self.apply_filter):
                    words.add(e.text)

        with open(self.out_word_list.get_path(), "w") as word_file:
            for w in sorted(words):
                word_file.write("%s\n" % w)


class FilterLexiconByWordListJob(Job):
    """
    Filter lemmata to given word list.
    Warning: case_sensitive parameter does the opposite. Kept for backwards-compatibility.
    """

    __sis_hash_exclude__ = {"check_synt_tok": False}

    def __init__(self, bliss_lexicon, word_list, case_sensitive=False, check_synt_tok=False):
        """
        :param tk.Path bliss_lexicon: lexicon file to be handeled
        :param tk.Path word_list: filter lexicon by this word list
        :param bool case_sensitive: filter lemmata case-sensitive. Warning: parameter does the opposite.
        :param bool check_synt_tok: keep also lemmata where the syntactic token matches word_list
        """
        self.set_vis_name("Filter Lexicon by Word List")

        self.bliss_lexicon = bliss_lexicon
        self.word_list = word_list
        self.case_sensitive = case_sensitive
        self.check_synt_tok = check_synt_tok

        self.out_bliss_lexicon = self.output_path(os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True)

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
            all_synt_tok = lemma.findall("./synt/tok")
            if any(
                transform(orth.text) in words
                or "special" in lemma.attrib
                or (orth.text is not None and orth.text.startswith("["))
                for orth in lemma.findall("orth")
            ) or (
                self.check_synt_tok
                and len(all_synt_tok) > 0
                and all([transform(tok.text) in words for tok in all_synt_tok])
            ):
                root.append(lemma)

        with uopen(self.out_bliss_lexicon.get_path(), "wt") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(root, "unicode"))


class LexiconUniqueOrthJob(Job):
    """Merge lemmata with the same orthography."""

    __sis_hash_exclude__ = {"merge_multi_orths_lemmata": False, "deduplicate_special_lemmata": False}

    def __init__(self, bliss_lexicon, merge_multi_orths_lemmata=False, deduplicate_special_lemmata=False):
        """
        :param tk.Path bliss_lexicon: lexicon file to be handeled
        :param bool merge_multi_orths_lemmata: if True, also lemmata containing
            multiple orths are merged based on their primary orth. Otherwise
            they are ignored.

            Merging strategy
            - orth/phon/eval
                all orth/phon/eval elements are merged together
            - synt
                synt element is only copied to target lemma when
                    1) the target lemma does not already have one
                    2) and the rest to-be-merged-lemmata have any synt
                       element.
                    ** having a synt <=> synt is not None
                this could lead to INFORMATION LOSS if there are several
                different synt token sequences in the to-be-merged lemmata
        :param bool deduplicate_special_lemmata: if True, special lemmata will also be considered
            in the unique process.
        """
        self.set_vis_name("Make Lexicon Orths Unique")

        self.bliss_lexicon = bliss_lexicon
        self.merge_multi_orths_lemmata = merge_multi_orths_lemmata
        self.deduplicate_special_lemmata = deduplicate_special_lemmata

        self.out_bliss_lexicon = self.output_path(os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        orth2lemmata = collections.defaultdict(list)

        for lemma in lex.lemmata:
            if lemma.special and not self.deduplicate_special_lemmata:
                continue
            num_orths = len(lemma.orth)
            if num_orths < 1:
                continue
            if num_orths > 1 and not self.merge_multi_orths_lemmata:
                continue
            orth2lemmata[lemma.orth[0]].append(lemma)

        for orth, lemmata in orth2lemmata.items():
            if len(lemmata) < 2:
                continue
            final_lemma = lemmata[0]
            for lemma in lemmata[1:]:
                for orth in lemma.orth:
                    if orth not in final_lemma.orth:
                        final_lemma.orth.append(orth)
                for phon in lemma.phon:
                    if phon not in final_lemma.phon:
                        final_lemma.phon.append(phon)
                if final_lemma.synt is None and lemma.synt is not None:
                    final_lemma.synt = lemma.synt
                for eval in lemma.eval:
                    if eval not in final_lemma.eval:
                        final_lemma.eval.append(eval)
                lex.lemmata.remove(lemma)

        write_xml(self.out_bliss_lexicon, element_tree=lex.to_xml())


class LexiconFromTextFileJob(Job):
    """
    Create a bliss lexicon from a regular text file, where each line contains:
    <WORD> <PHONEME1> <PHONEME2> ...
    separated by tabs or spaces.
    The lemmata will be added in the order they appear in the text file,
    the phonemes will be sorted alphabetically.
    Phoneme variants of the same word need to appear next to each other.

    WARNING: No special lemmas or phonemes are added,
    so do not use this lexicon with RASR directly!

    As the splitting is taken from RASR and not fully tested,
    it might not work in all cases so do not use this job
    without checking the output manually on new lexica.
    """

    def __init__(self, text_file, compressed=True):
        """
        :param Path text_file:
        :param compressed: save as .xml.gz
        """
        self.text_file = text_file

        self.out_bliss_lexicon = self.output_path("lexicon.xml.gz" if compressed else "lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()

        phonemes = set()
        seen_lemma = {}
        with uopen(self.text_file.get_path()) as f:
            for line in f:
                # splitting is taken from RASR
                # src/Tools/Bliss/blissLexiconLib.py#L185
                s = line.split(None, 1)
                orth = s[0].split("\\", 1)[0]
                phon_variants = [tuple(p.split()) for p in s[1].split("\\") if p.strip()]
                for phon_variant in phon_variants:
                    phonemes.update(phon_variant)
                phon = [" ".join(v) for v in phon_variants]
                if orth in seen_lemma:
                    lemma = seen_lemma[orth]
                    for p in phon:
                        if p not in lemma.phon:
                            lemma.phon.append(p)
                else:
                    lemma = lexicon.Lemma(orth=[orth], phon=phon)
                    seen_lemma[orth] = lemma
                    lex.add_lemma(lemma)

        for phoneme in sorted(phonemes):
            lex.add_phoneme(phoneme)

        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())


class GraphemicLexiconFromWordListJob(Job):
    default_transforms = {".": "DOT", "+": "PLUS", "{": "LBR", "}": "RBR"}

    def __init__(self, word_list_file, add_unknown=False, add_noise=False, transforms=None):
        self.add_unknown = add_unknown
        self.add_noise = add_noise
        self.transforms = transforms if transforms is not None else self.default_transforms
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

        # TODO: figure out requirements on synt/eval element for differnt types of lemmata
        # silence lemma, needs synt/eval element with empty token sequence
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["sil"],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
        # sentence border lemmata, needs no eval element
        lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE_BEGIN]"], synt=["<s>"], special="sentence-begin"))
        lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE_END]"], synt=["</s>"], special="sentence-end"))
        # unknown lemma, needs no synt/eval element
        if self.add_unknown:
            lex.add_lemma(lexicon.Lemma(orth=["[UNKNOWN]"], phon=["unk"], special="unknown"))
            # TODO: synt = ["<UNK>"] ???
        # noise lemma, needs empty synt token sequence but no eval element?
        if self.add_noise:
            lex.add_lemma(
                lexicon.Lemma(
                    orth=["[NOISE]"],
                    phon=["noise"],
                    synt=[],
                    special="unknown",
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


class SpellingConversionJob(Job):
    """Spelling conversion for lexicon."""

    __sis_hash_exclude__ = {"keep_original_target_lemmas": False}

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        orth_mapping_file: Union[str, tk.Path],
        mapping_file_delimiter: str = " ",
        mapping_rules: Optional[List[Tuple[str, str, str]]] = None,
        invert_mapping: bool = False,
        keep_original_target_lemmas: bool = False,
    ):
        """
        :param Path bliss_lexicon:
            input lexicon, whose lemmata all have unique PRIMARY orth
            to reach the above requirements apply LexiconUniqueOrthJob
        :param str|tk.Path orth_mapping_file:
            orthography mapping file: *.json *.json.gz *.txt *.gz
            in case of plain text file
                one can adjust mapping_delimiter
                a line starting with "#" is a comment line
        :param str mapping_file_delimiter:
            delimiter of source and target orths in the mapping file
            relevant only if mapping is provided with a plain text file
        :param Optional[List[Tuple[str, str, str]]] mapping_rules
            a list of mapping rules, each rule is represented by 3 strings
                (source orth-substring, target orth-substring, pos)
                where
                pos should be one of ["leading", "trailing", "any"]
            e.g. the rule ("zation", "sation", "trailing") will convert orth
            ending with -zation to orth ending with -sation
            set this ONLY when it's clearly defined rules which can not
            generate any kind of ambiguities
        :param bool invert_mapping:
            invert the input orth mapping
            NOTE: this also affects the pairs which are inferred from mapping_rules
         :param bool keep_original_target_lemmas:
            set this option to True if you want to keep the original target lemma in addition.
            This is needed if a LM contains both spelling variants and we want to clean but keep
            the usage of all LM probabilities.

        """
        self.set_vis_name("Convert Between Regional Orth Spellings")

        self.bliss_lexicon = bliss_lexicon
        self.orth_mapping_file = orth_mapping_file
        self.invert_mapping = invert_mapping
        self.mapping_file_delimiter = mapping_file_delimiter
        self.mapping_rules = mapping_rules
        self.keep_original_target_lemmas = keep_original_target_lemmas

        self.out_bliss_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _lemma_to_str(lemma, description):
        """Convert a lemma instance to a logging string
        :param Lemma lemma:
            the lemma to be converted to str representation
        :param str description:
            extra description for this lemma
        :return:
            str
        """
        xml_string = xml.dom.minidom.parseString(ET.tostring(lemma.to_xml())).toprettyxml(indent=" " * 2)
        lemma_str = "\n".join(xml_string.split("\n")[1:])
        lemma_str = description + "\n" + lemma_str
        return lemma_str

    def run(self):
        # load mapping from json or plain text file
        orth_map_file_str = tk.uncached_path(self.orth_mapping_file)
        is_json = orth_map_file_str.endswith(".json") | orth_map_file_str.endswith(".json.gz")
        if is_json:
            with uopen(orth_map_file_str, "rt") as f:
                mapping = json.load(f)
            if self.invert_mapping:
                mapping = {v: k for k, v in mapping.items()}
        else:
            mapping = dict()
            with uopen(orth_map_file_str, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    orths = line.split(self.mapping_file_delimiter)
                    if len(orths) != 2:
                        raise ValueError(
                            "The selected mapping delimiter is not valid, it "
                            "generates {} orths for line "
                            "'{}'!".format(len(orths), line)
                        )
                    source_orth, target_orth = orths
                    if self.invert_mapping:
                        source_orth, target_orth = target_orth, source_orth
                    mapping[source_orth] = target_orth
        num_mappings = len(mapping)
        logging.info("A total of {} word mapping pairs".format(num_mappings))

        # compile mapping patterns from extra mapping_rules
        mapping_patterns = []
        if self.mapping_rules:
            for sub_source, sub_target, pos in self.mapping_rules:
                if pos not in ["leading", "trailing", "any"]:
                    raise ValueError(
                        "position of a mapping rule must be one of "
                        "['leading', 'trailing', 'any'], got '{}' for rule "
                        "{} ==> {}.".format(pos, sub_source, sub_target)
                    )
                if self.invert_mapping:
                    sub_source, sub_target = sub_target, sub_source
                pattern = re.escape(sub_source)
                replacement = sub_target
                if pos == "leading":
                    pattern = r"^" + pattern + r"(\S{3,})$"
                    replacement = r"{}\1".format(sub_target)
                if pos == "trailing":
                    pattern = r"^(\S{3,})" + pattern + r"$"
                    replacement = r"\1{}".format(sub_target)
                pattern = re.compile(pattern, re.IGNORECASE)
                mapping_patterns.append((pattern, replacement))

        # load input lexicon and build "orth to lemma" dict
        # extend mapping dict if extra mapping_rules were defined
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        orth2lemma = {}
        for lemma in lex.lemmata:
            primary_orth = lemma.orth[0]
            if primary_orth in orth2lemma:
                raise ValueError(
                    "There shouldn't be two lemmata with the same primary "
                    "orth, apply LexiconUniqueOrthJob before doing spelling "
                    "conversion!"
                )
            orth2lemma[primary_orth] = lemma
            if primary_orth in mapping:
                continue
            for pattern, replacement in mapping_patterns:
                if pattern.search(primary_orth):
                    target_orth = pattern.sub(replacement, primary_orth)
                    mapping[primary_orth] = target_orth
                    logging.info(
                        "added mapping pair through mapping rule: {} ==> " "{}".format(primary_orth, target_orth)
                    )
                    break
        if len(mapping) > num_mappings:
            logging.info(
                "A total of {} mapping pairs added through extra mapping " "rules".format(len(mapping) - num_mappings)
            )

        # spelling conversion
        for source_orth, target_orth in mapping.items():
            if source_orth == target_orth:
                continue
            target_lemma = orth2lemma.get(target_orth, None)
            source_lemma = orth2lemma.get(source_orth, None)
            if target_lemma:
                logging.info(self._lemma_to_str(target_lemma, "target lemma"))
            else:
                logging.info("No target lemma for: {}".format(target_orth))
            if source_lemma:
                logging.info(self._lemma_to_str(source_lemma, "source lemma"))
            else:
                logging.info("No source lemma for: {}".format(source_orth))
            if target_lemma:
                if self.keep_original_target_lemmas:
                    copy_target_lemma = copy.deepcopy(target_lemma)
                if source_lemma:
                    for orth in source_lemma.orth:
                        if orth not in target_lemma.orth:
                            target_lemma.orth.append(orth)
                    for phon in source_lemma.phon:
                        if phon not in target_lemma.phon:
                            target_lemma.phon.append(phon)
                    for eval in source_lemma.eval:
                        if eval not in target_lemma.eval:
                            target_lemma.eval.append(eval)
                    if source_lemma in lex.lemmata:
                        if not self.keep_original_target_lemmas:
                            lex.lemmata.remove(source_lemma)
                        else:
                            # Replace the source lemma and keep original target lemma as well
                            # without changing the position in the lexicon
                            source_position = lex.lemmata.index(source_lemma)
                            target_position = lex.lemmata.index(target_lemma)
                            lex.lemmata[source_position] = target_lemma
                            lex.lemmata[target_position] = copy_target_lemma
                if not target_lemma.synt:
                    if source_lemma and source_lemma.synt:
                        target_lemma.synt = source_lemma.synt
                    else:
                        target_lemma.synt = source_orth.split()
                if self.keep_original_target_lemmas and not source_lemma:
                    target_position = lex.lemmata.index(target_lemma)
                    lex.lemmata.insert(target_position - 1, copy_target_lemma)
                logging.info(self._lemma_to_str(target_lemma, "final lemma"))
            elif source_lemma:
                source_lemma.orth.insert(0, target_orth)
                if not source_lemma.synt:
                    source_lemma.synt = source_orth.split()
                logging.info(self._lemma_to_str(source_lemma, "final lemma"))
            logging.info("-" * 60)

        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())
