__all__ = ["ApplyBPEModelToLexiconJob", "ApplyBPEToTextJob"]

import subprocess as sp
import os
import shutil
import sys
import tempfile
from typing import Optional
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

from i6_core.lib.lexicon import Lexicon
import i6_core.util as util


class ApplyBPEModelToLexiconJob(Job):
    """
    Apply BPE codes to a Bliss lexicon file
    """

    def __init__(self, bliss_lexicon, bpe_codes, bpe_vocab=None, subword_nmt_repo=None):
        """
        :param Path bliss_lexicon:
        :param Path bpe_codes:
        :param Path|None bpe_vocab:
        :param Optional[Path] subword_nmt_repo:
        """
        self.bliss_lexicon = bliss_lexicon
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)

        self.out_converted_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon_path = self.bliss_lexicon.get_path()

        lexicon = Lexicon()
        lexicon.load(lexicon_path)

        lm_tokens = set()
        for l in lexicon.lemmata:
            for orth in l.orth:
                lm_tokens.add(orth)
            for token in l.synt or []:  # l.synt can be None
                lm_tokens.add(token)
            for eval in l.eval:
                for t in eval:
                    lm_tokens.add(t)

        lm_tokens = list(lm_tokens)

        with util.uopen("words", "wt") as f:
            for t in lm_tokens:
                f.write("%s\n" % t)

        apply_binary = self.subword_nmt_repo.join_right("subword_nmt/apply_bpe.py")
        args = [
            sys.executable,
            apply_binary.get_path(),
            "--input",
            "words",
            "--codes",
            self.bpe_codes.get_path(),
            "--output",
            "bpes",
        ]
        if self.bpe_vocab is not None:
            args += ["--vocabulary", self.bpe_vocab.get_path()]
        sp.run(args, check=True)

        with util.uopen("bpes", "rt") as f:
            bpe_tokens = [l.strip().split() for l in f]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        for l in lexicon.lemmata:
            if l.special is None and len(l.orth) > 0:
                if not l.synt and len(l.eval) == 0:
                    o = l.orth[0]
                    l.synt = w2b[o]
                    l.eval.append([o])
                if l.synt:
                    l.synt = sum([w2b[token] for token in l.synt], [])
                if len(l.eval) > 0:
                    l.eval = [
                        sum([w2b[t] for t in token_sequence], [])
                        for token_sequence in l.eval
                    ]

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        with util.uopen(self.out_converted_lexicon.get_path(), "wb") as f:
            tree.write(f, encoding="utf-8")


class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
        self,
        text_file: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: Optional[tk.Path] = None,
        subword_nmt_repo: Optional[tk.Path] = None,
        gzip_output: bool = False,
        mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param bpe_codes: bpe codes file, e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param bpe_vocab: if provided, then merge operations that produce OOV are reverted,
            use e.g. ReturnnTrainBpeJob.out_bpe_dummy_count_vocab
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param gzip_output: use gzip on the output text
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        self.gzip_output = gzip_output

        self.out_bpe_text = self.output_path(
            "words_to_bpe.txt.gz" if gzip_output else "words_to_bpe.txt"
        )

        self.mini_task = mini_task
        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            input_file = self.text_file.get_path()
            tmp_infile = os.path.join(tmp, "in_text.txt")
            tmp_outfile = os.path.join(tmp, "out_text.txt")
            with util.uopen(tmp_infile, "wt") as out:
                sp.call(["zcat", "-f", input_file], stdout=out)
            cmd = [
                sys.executable,
                os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py"),
                "--input",
                tmp_infile,
                "--codes",
                self.bpe_codes.get_path(),
                "--output",
                tmp_outfile,
            ]

            if self.bpe_vocab:
                cmd += ["--vocabulary", self.bpe_vocab.get_path()]

            util.create_executable("apply_bpe.sh", cmd)
            sp.run(cmd, check=True)

            if self.gzip_output:
                with util.uopen(tmp_outfile, "rt") as fin, util.uopen(
                    self.out_bpe_text, "wb"
                ) as fout:
                    sp.call(["gzip"], stdin=fin, stdout=fout)
            else:
                shutil.copy(tmp_outfile, self.out_bpe_text.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)
