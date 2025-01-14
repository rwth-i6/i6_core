__all__ = ["TrainBPEModelJob", "ReturnnTrainBpeJob"]

import subprocess as sp
import os
import sys
from typing import Optional

from sisyphus import *

import i6_core.util as util

Path = setup_path(__package__)


class TrainBPEModelJob(Job):
    """
    Create a bpe codes file using the official subword-nmt repo, either installed from pip
    or https://github.com/rsennrich/subword-nmt

    This job is deprecated, to create BPE codes that are compatible with legacy (non-sisyphus) RETURNN setups
    using e.g. language models from Kazuki, please use the `ReturnnTrainBpeJob`.

    Otherwise, please consider using the `sentencepiece` implementation.
    """

    def __init__(
        self,
        text_corpus: tk.Path,
        symbols: int = 1000,
        min_frequency: int = 2,
        dict_input: bool = False,
        total_symbols: bool = False,
        subword_nmt_repo: Optional[tk.Path] = None,
    ):
        """
        :param text_corpus: text corpus path.
        :param symbols: number of symbols.
        :param min_frequency: mimumu frequency of a symbol.
        :param dict_input: input file will be interpreted as a dict.
        :param total_symbols: this param is not set in the `learn_bpe.py`.
        :param subword_nmt_repo: path to subword_nmt repo.
        """
        self.text_corpus = text_corpus
        self.symbols = symbols
        self.min_frequency = min_frequency
        self.dict_input = dict_input
        self.total_symbols = total_symbols

        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)

        self.out_code_file = self.output_path("codes")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        train_binary = self.subword_nmt_repo.join_right("subword_nmt/learn_bpe.py")
        args = [
            sys.executable,
            train_binary.get_path(),
            "-o",
            self.out_code_file.get_path(),
            "-s",
            str(self.symbols),
            "--min-frequency",
            str(self.min_frequency),
        ]
        if self.dict_input:
            args += ["--dict-input"]
        if self.total_symbols:
            args += ["--total-symbols"]

        text_corpus = self.text_corpus.get_path()

        with util.uopen(text_corpus, "rb") as f:
            p = sp.Popen(args, stdin=sp.PIPE, stdout=sys.stdout, stderr=sys.stderr)
            while True:
                data = f.read(4096)
                if len(data) > 0:
                    p.stdin.write(data)
                else:
                    break

            p.communicate()
            assert p.returncode == 0


class ReturnnTrainBpeJob(Job):
    __sis_hash_exclude__ = {"allow_special_labels": False}

    """
    Create Bpe codes and vocab files compatible with RETURNN BytePairEncoding
    Repository:
        https://github.com/rwth-i6/subword-nmt

    This job can be used to produce BPE codes compatible to legacy (non-sisyphus) RETURNN setups.

    Outputs:
        - bpe_codes: the codes file to apply BPE to any text
        - bpe_vocab: the index vocab in the form of {"<token>": <index>, ...} that can be used e.g. for RETURNN
            Will contain <s> and </s> pointing to index 0 and the unk_label pointing to index 1
        - bpe_dummy_count_vocab: a text file containing all words, to be used with the `ApplyBPEToTextJob`
            DOES NOT INCLUDE COUNTS, but just set each count to -1. Is used to not cause invalid merges
            when converting text to the BPE form.
        - vocab_size: variable containing the number of indices
    """

    def __init__(
        self,
        text_file: tk.Path,
        bpe_size: int,
        unk_label: str = "UNK",
        subword_nmt_repo: Optional[tk.Path] = None,
        allow_special_labels: bool = False,
    ):
        """
        :param text_file: corpus text file, .gz compressed or uncompressed
        :param bpe_size: number of BPE merge operations
        :param unk_label: unknown label
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param allow_special_labels: allows special labels during vocab creation.
        """
        self.text_file = text_file
        self.bpe_size = bpe_size
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        self.unk_label = unk_label
        self.allow_special_labels = allow_special_labels

        self.out_bpe_codes = self.output_path("bpe.codes")
        self.out_bpe_vocab = self.output_path("bpe.vocab")
        self.out_bpe_dummy_count_vocab = self.output_path("bpe.dummy_count.vocab")
        self.out_vocab_size = self.output_var("vocab_size")

    def run(self):
        bpe_codes_cmd = [
            sys.executable,
            os.path.join(self.subword_nmt_repo.get_path(), "learn_bpe.py"),
            "--output",
            self.out_bpe_codes.get_path(),
            "--symbols",
            str(self.bpe_size),
        ]

        util.create_executable("create_bpe_codes.sh", bpe_codes_cmd)

        with util.uopen(self.text_file, "rb") as f:
            p = sp.Popen(bpe_codes_cmd, stdin=sp.PIPE, stdout=sys.stdout, stderr=sys.stderr)
            while True:
                data = f.read(4096)
                if len(data) > 0:
                    p.stdin.write(data)
                else:
                    break
            p.communicate()
            assert p.returncode == 0

        bpe_vocab_cmd = [
            sys.executable,
            os.path.join(self.subword_nmt_repo.get_path(), "create-py-vocab.py"),
            "--txt",
            self.text_file.get_path(),
            "--bpe",
            self.out_bpe_codes.get_path(),
            "--unk",
            self.unk_label,
            "--out",
            self.out_bpe_vocab.get_path(),
        ]
        if self.allow_special_labels:
            bpe_vocab_cmd += ["--allow_special_labels"]

        util.create_executable("create_bpe_vocab.sh", bpe_vocab_cmd)
        sp.run(bpe_vocab_cmd, check=True)

        with util.uopen(self.out_bpe_vocab) as f, util.uopen(self.out_bpe_dummy_count_vocab, "wt") as txt_vocab:
            vocab = eval(f.read())
            num_labels = max(list(vocab.values())) + 1  # 0-based index
            self.out_vocab_size.set(num_labels)
            for l in vocab.keys():
                txt_vocab.write(f"{l} -1\n")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 4})
