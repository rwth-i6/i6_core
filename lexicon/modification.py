__all__ = ["WriteLexiconJob", "MergeLexiconJob", "AddEowPhonemesToLexiconJob"]

import copy
from collections import OrderedDict, defaultdict
import itertools
from typing import Generator, List, Optional, Set

from sisyphus import Job, Task, tk

from i6_core.lib import lexicon
from i6_core.util import write_xml


class WriteLexiconJob(Job):
    """
    Create a bliss lexicon file from a static Lexicon.

    Supports optional sorting of phonemes and lemmata.

    Example for a static lexicon:

    .. code: python

        static_lexicon = lexicon.Lexicon()
        static_lexicon.add_lemma(
            static_lexiconicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["[SILENCE]"],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
        # set synt and eval carefully
        # synt == None   --> nothing                 no synt element
        # synt == []     --> "<synt />"              meant to be empty synt token sequence
        # synt == [""]   --> "<synt><tok /></synt>"  incorrent
        # eval == []     --> nothing                 no eval element
        # eval == [[]]   --> "<eval />"              meant to be empty eval token sequence
        # eval == [""]   --> "<eval />"              equivalent to [[]], but not encouraged
        # eval == [[""]] --> "<eval><tok /></eval>"  incorrect
        static_lexicon.add_lemma(
            static_lexiconicon.Lemma(
                orth=["[UNKNOWN]"],
                phon=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )
        static_lexicon.add_phoneme("[SILENCE]", variation="none")
        static_lexicon.add_phoneme("[UNKNOWN]", variation="none")
    """

    def __init__(self, static_lexicon, sort_phonemes=False, sort_lemmata=False, compressed=True):
        """
        :param lexicon.Lexicon static_lexicon: A Lexicon object
        :param bool sort_phonemes: sort phoneme inventory alphabetically
        :param bool sort_lemmata: sort lemmata alphabetically based on first orth entry
        :param bool compressed: compress final lexicon
        """
        self.static_lexicon = static_lexicon
        self.sort_phonemes = sort_phonemes
        self.sort_lemmata = sort_lemmata

        self.out_bliss_lexicon = self.output_path("lexicon.xml.gz" if compressed else "lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        if self.sort_phonemes:
            sorted_phoneme_list = [
                (k, self.static_lexicon.phonemes[k]) for k in sorted(self.static_lexicon.phonemes.keys())
            ]
            for phoneme_tuple in sorted_phoneme_list:
                lex.add_phoneme(symbol=phoneme_tuple[0], variation=phoneme_tuple[1])
        else:
            lex.phonemes = self.static_lexicon.phonemes

        if self.sort_lemmata:
            lemma_dict = {}
            for lemma in self.static_lexicon.lemmata:
                # sort by first orth entry
                lemma_dict[lemma.orth[0]] = lemma
            lex.lemmata = [lemma_dict[key] for key in sorted(lemma_dict.keys())]
        else:
            lex.lemmata = self.static_lexicon.lemmata

        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())

    @classmethod
    def _fix_hash_for_lexicon(cls, new_lexicon):
        """
        The "old" lexicon had an incorrect "synt" type, after fixing
        the hashes for the lexicon changed, so this job here
        needs to revert the lexicon to the old "synt" type.

        :param lexicon.Lexicon new_lexicon:
        :return: lexicon in the legacy format
        :type: lexicon.Lexicon
        """
        lex = lexicon.Lexicon()
        lex.phonemes = new_lexicon.phonemes
        lex.lemmata = []
        for new_lemma in new_lexicon.lemmata:
            lemma = copy.deepcopy(new_lemma)
            lemma.synt = [new_lemma.synt] if new_lemma.synt is not None else []
            lex.lemmata.append(lemma)

        return lex

    @classmethod
    def hash(cls, parsed_args):
        parsed_args = parsed_args.copy()
        parsed_args["static_lexicon"] = cls._fix_hash_for_lexicon(parsed_args["static_lexicon"])
        return super().hash(parsed_args)


class MergeLexiconJob(Job):
    """
    Merge multiple bliss lexica into a single bliss lexicon.

    Phonemes and lemmata can be individually sorted alphabetically or kept as is.

    When merging a lexicon with a static lexicon, putting the static lexicon first
    and only sorting the phonemes will result in the "typical" lexicon structure.

    Please be aware that the sorting or merging of lexica that were already used
    will create a new lexicon that might be incompatible to previously generated alignments.
    """

    __sis_hash_exclude__ = {"deduplicate_lemmata": False}

    def __init__(
        self, bliss_lexica, sort_phonemes=False, sort_lemmata=False, compressed=True, deduplicate_lemmata=False
    ):
        """
        :param list[Path] bliss_lexica: list of bliss lexicon files (plain or gz)
        :param bool sort_phonemes: sort phoneme inventory alphabetically
        :param bool sort_lemmata: sort lemmata alphabetically based on first orth entry
        :param bool compressed: compress final lexicon
        :param bool deduplicate_lemmata: whether to deduplicate lemmatas, only applied when sort_lemmata=True
        """
        self.lexica = bliss_lexica
        self.sort_phonemes = sort_phonemes
        self.sort_lemmata = sort_lemmata
        self.deduplicate_lemmata = deduplicate_lemmata

        self.out_bliss_lexicon = self.output_path("lexicon.xml.gz" if compressed else "lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged_lex = lexicon.Lexicon()

        lexica = []
        for lexicon_path in self.lexica:
            lex = lexicon.Lexicon()
            lex.load(lexicon_path.get_path())
            lexica.append(lex)

        # combine the phonemes
        merged_phonemes = OrderedDict()
        for lex in lexica:
            for symbol, variation in lex.phonemes.items():
                if symbol in merged_phonemes.keys():
                    assert variation == merged_phonemes[symbol], "conflicting phoneme variant for phoneme: %s" % symbol
                else:
                    merged_phonemes[symbol] = variation

        if self.sort_phonemes:
            sorted_phoneme_list = [(k, merged_phonemes[k]) for k in sorted(merged_phonemes.keys())]
            for phoneme_tuple in sorted_phoneme_list:
                merged_lex.add_phoneme(symbol=phoneme_tuple[0], variation=phoneme_tuple[1])
        else:
            merged_lex.phonemes = merged_phonemes

        # combine the lemmata
        if self.sort_lemmata:
            lemma_dict = defaultdict(list)
            for lex in lexica:
                for lemma in lex.lemmata:
                    # sort by first orth entry
                    orth_key = lemma.orth[0] if lemma.orth else ""
                    if self.deduplicate_lemmata:
                        # don't add the lemma when there's already an equal lemma
                        if len(lemma_dict[orth_key]) > 0 and lemma == lemma_dict[orth_key][0]:
                            continue
                    lemma_dict[orth_key].append(lemma)
            merged_lex.lemmata = list(itertools.chain(*[lemma_dict[key] for key in sorted(lemma_dict.keys())]))
        else:
            for lex in lexica:
                # check for existing orths to avoid overlap
                merged_lex.lemmata.extend(lex.lemmata)

        write_xml(self.out_bliss_lexicon.get_path(), merged_lex.to_xml())


class AddEowPhonemesToLexiconJob(Job):
    def __init__(self, bliss_lexicon: tk.Path, nonword_phones: Optional[List] = None, boundary_marker: str = "#"):
        """
        Extends phoneme set of a lexicon by additional end-of-word (eow) versions
        of all regular phonemes. Modifies lemmata to use the new eow-version
        of the final phoneme in each pronunciation.

        :param bliss_lexicon: Base lexicon to be modified.
        :param nonword_phones: List of nonword-phones for which no eow-versions will be added, e.g. [noise].
                               Phonemes that occur in special lemmata are found automatically and do not need
                               to be specified here.
        :param boundary_marker: String that is appended to phoneme symbols to mark eow-version.
        """
        self.bliss_lexicon = bliss_lexicon
        self.nonword_phones = nonword_phones or []
        self.boundary_marker = boundary_marker

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _eow_phoneme(self, phoneme: str) -> str:
        """
        Creates the eow-version of a given phoneme.
        """
        return phoneme + self.boundary_marker

    def _modify_pronunciation(self, pronunciation: str, special_phonemes: Set[str]) -> str:
        """
        Find the rightmost phoneme in the pronunciation that is not special and replace it by its
        eow-version. Might do nothing if all phonemes are special.

        Example: "AA BB [noise]" -> "AA BB# [noise]"

        :param pronunciation: Original pronunciation as a string containing phonemes separated by whitespaces
        :param special_phonemes: Set of special phonemes that should be skipped over.
        :return: Modified pronunciation
        """
        phoneme_list = pronunciation.split()

        for i, phoneme in reversed(list(enumerate(phoneme_list))):
            if phoneme in special_phonemes:
                continue
            phoneme_list[i] = self._eow_phoneme(phoneme)
            break

        return " ".join(phoneme_list)

    def run(self):
        in_lex = lexicon.Lexicon()
        in_lex.load(self.bliss_lexicon.get_path())

        out_lex = lexicon.Lexicon()

        # Identify all 'special' phonemes in the given lexicon. A phoneme is
        # deemed 'special' if it appears in the pronunciation of a special lemma.
        # This should identify e.g. 'silence', 'blank', 'unknown' etc.
        special_phonemes = set()
        for lemma in in_lex.lemmata:
            if lemma.special is None:
                continue
            for phon in lemma.phon:
                special_phonemes.update(phon.split())
        special_phonemes.update(self.nonword_phones)

        # Add all phonemes to out_lexicon and create list of eow-phonemes
        eow_phonemes = []
        for phoneme, variation in in_lex.phonemes.items():
            out_lex.add_phoneme(phoneme, variation)
            if phoneme not in special_phonemes:
                eow_phonemes.append((self._eow_phoneme(phoneme), variation))

        # Add eow-phonemes to out_lexicon
        for eow_phoneme, variation in eow_phonemes:
            out_lex.add_phoneme(eow_phoneme, variation)

        # Modify lemmata to include eow-phoneme
        for lemma in in_lex.lemmata:
            lemma.phon = [self._modify_pronunciation(phon, special_phonemes) for phon in lemma.phon]
            out_lex.add_lemma(lemma)

        # Write resulting lexicon to file
        write_xml(self.out_lexicon.get_path(), out_lex.to_xml())
