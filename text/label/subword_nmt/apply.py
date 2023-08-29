__all__ = ["ApplyBPEModelToLexiconJob", "ApplyBPEToTextJob"]

import os
import sys
import shutil
import tempfile
import subprocess as sp
from typing import Optional
import xml.etree.ElementTree as ET

from sisyphus import *

import i6_core.util as util
from i6_core.lib.lexicon import Lexicon, Lemma


class ApplyBPEModelToLexiconJob(Job):
    """
    Apply BPE codes on a Bliss lexicon file
    """

    def __init__(
        self,
        base_lexicon_path: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: Optional[tk.Path] = None,
        unk_label: str = "UNK",
        add_silence: bool = True,
        add_other_special: bool = False,
    ):
        """
        :param tk.Path base_lexicon_path: path to a Bliss lexicon
        :param tk.Path bpe_codes: path to BPE codes file, use e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param tk.Path bpe_vocab: path to BPE vocab file used to revert merge operations that produce OOV,
            use e.g. ReturnnTrainBPEJob.out_bpe_vocab;
        :param tk.Path|None subword_nmt_repo: path to subword nmt repository, see also `CloneGitRepositoryJob`
        :param str unk_label:
        :param bool add_silence: explicitly include a [SILENCE] phoneme and lemma
        :param bool add_other_special: explicitly include special lemmata from base_lexicon_path
        """
        self.base_lexicon_path = base_lexicon_path
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        self.unk_label = unk_label
        self.add_silence = add_silence
        self.add_other_special = add_other_special

        self.out_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lm_tokens = set()
        other_special = []

        base_lexicon = Lexicon()
        base_lexicon.load(self.base_lexicon_path)

        for l in base_lexicon.lemmata:
            if l.special:
                if l.special not in ["silence", "unknown"]:
                    other_special.append(l)
                continue
            for orth in l.orth or []:  # l.orth can be None
                lm_tokens.add(orth)
            for token in l.synt or []:  # l.synt can be None
                lm_tokens.add(token)
            for eval in l.eval or []:  # l.eval can be None
                for t in eval:
                    lm_tokens.add(t)

        lm_tokens = [lt for lt in lm_tokens if lt != ""]  # catch empty orth, e.g. '' for [SILENCE]

        with util.uopen("words.txt", "wt") as f:
            for t in lm_tokens:
                f.write(f"{t}\n")

        vocab = set()
        lexicon = Lexicon()

        lexicon.add_phoneme(self.unk_label, variation="none")

        if self.add_silence:
            lexicon.add_phoneme("[SILENCE]", variation="none")

        with util.uopen(self.bpe_vocab.get_path(), "rt") as bpe_vocab_file:
            with util.uopen("fake_count_vocab.txt", "wt") as fake_count_file:
                for line in bpe_vocab_file:
                    if "{" in line or "<s>" in line or "</s>" in line or "}" in line:
                        continue
                    symbol = line.split(":")[0][1:-1]
                    if symbol != self.unk_label:
                        fake_count_file.write(symbol + " -1\n")
                        symbol = symbol.replace(".", "_")
                        vocab.add(symbol)
                        lexicon.add_phoneme(symbol)

        apply_binary = os.path.join(tk.uncached_path(self.subword_nmt_repo), "apply_bpe.py")
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words.txt",
            "--codes",
            self.bpe_codes.get_path(),
            "--vocabulary",
            "fake_count_vocab.txt",
            "--output",
            "bpes.txt",
        ]
        sp.run(args, check=True)

        with util.uopen("bpes.txt", "rt") as f:
            bpe_tokens = [l.strip() for l in f]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        lexicon.add_lemma(Lemma(["[UNKNOWN]"], [self.unk_label], None, None, special="unknown"))

        if self.add_silence:
            lexicon.add_lemma(Lemma(["[SILENCE]"], ["[SILENCE]"], [], [[]], special="silence"))

        if self.add_other_special:
            for l in other_special:
                lexicon.add_lemma(l)

        for w, b in w2b.items():
            b = " ".join([token if token in vocab else self.unk_label for token in b.split()])
            lexicon.add_lemma(Lemma([w], [b.replace(".", "_")]))

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        util.write_xml(self.out_lexicon.get_path(), tree)


class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
        self,
        words_file: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: Optional[tk.Path] = None,
        gzip_output: bool = False,
        mini_task: bool = True,
    ):
        """
        :param tk.Path text_file: path to a words text file
        :param tk.Path bpe_codes: path to BPE codes file, use e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param tk.Path bpe_vocab: path to BPE vocab file used to revert merge operations that produce OOV,
            use e.g. ReturnnTrainBPEJob.out_bpe_vocab;
        :param tk.Path/None subword_nmt_repo: path to subword nmt repository , see also `CloneGitRepositoryJob`
        :param bool gzip_output: use gzip on the output text
        :param bool mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.words_file = words_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        self.gzip_output = gzip_output

        self.out_bpe_text = self.output_path("words_to_bpe.txt.gz" if gzip_output else "words_to_bpe.txt")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            words_file = self.words_file.get_path()
            tmp_outfile = os.path.join(tmp, "out_text.txt")

            with util.uopen(self.bpe_vocab.get_path(), "rt") as bpe_vocab_file:
                with util.uopen("fake_count_vocab.txt", "wt") as fake_count_file:
                    for line in bpe_vocab_file:
                        if "{" in line or "<" in line or "[" in line or "]" in line or ">" in line or "}" in line:
                            continue
                        symbol = line.split(":")[0][1:-1]
                        fake_count_file.write(symbol + " -1\n")

            apply_binary = os.path.join(tk.uncached_path(self.subword_nmt_repo), "apply_bpe.py")
            cmd = [
                sys.executable,
                apply_binary,
                "--input",
                words_file,
                "--codes",
                self.bpe_codes.get_path(),
                "--vocabulary",
                "fake_count_vocab.txt",
                "--output",
                tmp_outfile,
            ]
            util.create_executable("apply_bpe.sh", cmd)
            sp.run(cmd, check=True)

            if self.gzip_output:
                with util.uopen(tmp_outfile, "rt") as fin, util.uopen(self.out_bpe_text, "wb") as fout:
                    sp.call(["gzip"], stdin=fin, stdout=fout)
            else:
                shutil.copy(tmp_outfile, self.out_bpe_text.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)