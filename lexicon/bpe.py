__all__ = ["CreateBPELexiconJob"]

import subprocess as sp
import logging
import os
import sys
from typing import List, Optional, Set, Union
import xml.etree.ElementTree as ET

from sisyphus import Job, Task, tk

from i6_core.lib.lexicon import Lexicon, Lemma
import i6_core.util as util


class CreateBPELexiconJob(Job):
    """
    In a Bliss lexicon replace the phonetic representation with a BPE decomposition of the words that can be used e.g, for lexicon constrained BPE search.

    This job is still in experimental state, and only tested with Flashlight BPE decoding
    """

    __sis_hash_exclude__ = {"skip_unk_lemmas": False, "add_all_bpe_phonemes": True, "additional_words": None}

    def __init__(
        self,
        base_lexicon_path: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: tk.Path,
        unk_label: str = "UNK",
        vocab_blacklist: Optional[Union[List[str], Set[str]]] = None,
        keep_special_lemmas: bool = True,
        skip_unk_lemmas: bool = False,
        add_all_bpe_phonemes: bool = True,
        additional_words: Optional[tk.Path] = None,
    ):
        """
        :param base_lexicon_path: base lexicon (can be phoneme based) to take the lemmas from
        :param bpe_codes: bpe codes from the ReturnnTrainBPEJob
        :param bpe_vocab: vocab file to limit which bpe splits can be created
        :param subword_nmt_repo: cloned repository
        :param unk_label: unknown label, used in case a BPE token is created that is not in the vocab.
        :param vocab_blacklist: which bpe_vocab entries not to load into the "phoneme/bpe-token" inventory
            e.g. remove "<s>" and "</s>"
        :param keep_special_lemmas: If special lemmas should be kept,
            usually yes for RASR search and no for Flashlight search.
            The phonemes of the special lemmas will also be kept, therefore
            make sure there is no overlap with the BPE vocab.
        :param skip_unk_lemmas: whether simply skip lemmas out of the BPE vocab
            useful if you set vocab_blacklist
        :param add_all_bpe_phonemes: If set to True, all BPE vocab will be added to lexicon phonemes,
            otherwise, only phonemes appear in lexicon lemma will be added to the lexicon.
        :param additional_words: Aside from vocab specified in base_lexicon, we might want to convert some other words,
            e.g. untranslatable words by a g2p model in case of g2p-augmented lexicon
        """
        self.base_lexicon_path = base_lexicon_path
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo
        self.unk_label = unk_label
        if vocab_blacklist is None:
            self.vocab_blacklist = set()
        else:
            # convert list to set for faster "in" check
            self.vocab_blacklist = set(vocab_blacklist)
        self.keep_special_lemmas = keep_special_lemmas
        self.skip_unk_lemmas = skip_unk_lemmas
        self.add_all_bpe_phonemes = add_all_bpe_phonemes
        self.additional_words = additional_words

        self.out_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def _fill_lm_tokens(self, base_lexicon: Lexicon):
        lm_tokens = set()
        special_lemmas = []
        for lemma in base_lexicon.lemmata:
            if lemma.special is None:
                lm_tokens.update(lemma.orth)
            else:
                special_lemmas.append(lemma)

        return sorted(lm_tokens), special_lemmas

    def _fill_vocab_and_lexicon(self):
        lexicon = Lexicon()
        vocab = set()
        with util.uopen(self.bpe_vocab.get_path(), "rt") as f, util.uopen("fake_count_vocab.txt", "wt") as vocab_file:
            for line in f:
                line = line.strip()
                if line == "{" or line == "}":
                    continue
                # a line is e.g. '"phon": 0,' and we want to get 'phon' only
                symbol = line.split(":")[0][1:-1]
                if symbol not in self.vocab_blacklist:
                    # Fake count vocab filled with -1 so that all merges possible are done
                    vocab_file.write(symbol + " -1\n")
                    symbol = symbol.replace(".", "_")
                    vocab.add(symbol)
                    if self.add_all_bpe_phonemes:
                        lexicon.add_phoneme(symbol.replace(".", "_"))

        return vocab, lexicon

    def _fill_additional_words(self):
        additional_words_list = set()
        if self.additional_words is not None:
            with util.uopen(self.additional_words.get_path(), "rt") as f:
                for line in f:
                    line = line.strip()
                    additional_words_list.add(line)
        return sorted(additional_words_list)

    def run(self):
        base_lexicon = Lexicon()
        base_lexicon.load(self.base_lexicon_path)

        additional_words_list = self._fill_additional_words()
        for w in additional_words_list:
            base_lexicon.add_lemma(Lemma([w], None))  # add empty lemmata with only orth for additional words
        lm_tokens, special_lemmas = self._fill_lm_tokens(base_lexicon)

        with util.uopen("words", "wt") as f:
            for t in lm_tokens:
                f.write(f"{t}\n")

        vocab, lexicon = self._fill_vocab_and_lexicon()

        # add special lemmas back to lexicon
        if self.keep_special_lemmas is True:
            for special_lemma in special_lemmas:
                for pronunciation_variant in special_lemma.phon:
                    for phoneme in pronunciation_variant.split():
                        lexicon.add_phoneme(phoneme, variation=base_lexicon.phonemes[phoneme])
                lexicon.add_lemma(special_lemma)

        apply_binary = os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py")
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words",
            "--codes",
            self.bpe_codes.get_path(),
            "--vocabulary",
            "fake_count_vocab.txt",
            "--output",
            "bpes",
        ]
        sp.run(args, check=True)

        with util.uopen("bpes", "rt") as bpe_file:
            bpe_tokens = [line.strip() for line in bpe_file]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        used_vocab = set()
        for lemma in base_lexicon.lemmata:
            if lemma.special:
                continue
            for orth in lemma.orth:
                bpe_pron = " ".join([token if token in vocab else self.unk_label for token in w2b[orth].split()])
                if self.skip_unk_lemmas and self.unk_label in bpe_pron.split():
                    logging.info(f"Lemma {orth} is skipped due to unknown BPE vocab.")
                    continue
                used_vocab.update(set(bpe_pron.split()))
                lexicon.add_lemma(Lemma([orth], [bpe_pron.replace(".", "_")], lemma.synt, lemma.eval))

        if not self.add_all_bpe_phonemes:
            for symbol in sorted(used_vocab):
                lexicon.add_phoneme(symbol.replace(".", "_"))

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        util.write_xml(self.out_lexicon.get_path(), tree)
