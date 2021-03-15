__all__ = ["TrainBPEModelJob", "ReturnnTrainBpeJob"]

import gzip
import subprocess as sp
import os
import sys

from sisyphus import *

Path = setup_path(__package__)

from recipe.i6_asr.git import *


class TrainBPEModelJob(Job):
    def __init__(
        self,
        text_corpus,
        symbols=1000,
        min_frequency=2,
        dict_input=False,
        total_symbols=False,
    ):
        self.text_corpus = text_corpus
        self.symbols = symbols
        self.min_frequency = min_frequency
        self.dict_input = dict_input
        self.total_symbols = total_symbols

        self.subword_nmt_repo = CloneGitRepository(
            "https://github.com/rsennrich/subword-nmt.git"
        ).repository

        self.code_file = self.output_path("code")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        train_binary = os.path.join(
            self.subword_nmt_repo.get_path(), "subword_nmt/learn_bpe.py"
        )
        args = [
            sys.executable,
            train_binary,
            "-o",
            self.code_file.get_path(),
            "-s",
            str(self.symbols),
            "--min-frequency",
            str(self.min_frequency),
        ]
        if self.dict_input:
            args += ["--dict-input"]
        if self.total_symbols:
            args += ["--total-symbols"]

        text_corpus = tk.uncached_path(self.text_corpus)
        open_fun = gzip.open if text_corpus.endswith(".gz") else open

        with open_fun(text_corpus, "rb") as f:
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
    def __init__(self, text_file, bpe_size, unk_label="UNK"):
        """
        Create Bpe codes and vocab files
        NOTE: This uses Albert's subword-nmt fork which is compatible to RETURNN BytePairEncoding class

        :param Path text_file: corpus text file
        :param int bpe_size: number of BPE merge operations
        """
        self.text_file = text_file
        self.bpe_size = bpe_size
        self.unk_label = unk_label

        self.bpe_codes = self.output_path("bpe.codes")
        self.bpe_vocab = self.output_path("bpe.vocab")
        self.vocab_size = self.output_var("vocab_size")

        self.subword_nmt_repo = CloneGitRepository(
            "https://github.com/albertz/subword-nmt.git"
        ).repository

    def run(self):
        text_file = tk.uncached_path(self.text_file)
        cmd_cat_txt = "%s %s" % (
            "zcat" if text_file.endswith(".gz") else "cat",
            text_file,
        )
        self.sh(
            "%s | %s %s/learn_bpe.py --output %s --symbols %s"
            % (
                cmd_cat_txt,
                sys.executable,
                self.subword_nmt_repo.get_path(),
                self.bpe_codes.get_path(),
                self.bpe_size,
            )
        )
        self.sh(
            "%s %s/create-py-vocab.py --txt %s --bpe %s --unk %s --out %s"
            % (
                sys.executable,
                self.subword_nmt_repo.get_path(),
                text_file,
                self.bpe_codes.get_path(),
                self.unk_label,
                self.bpe_vocab.get_path(),
            )
        )

        with open(self.bpe_vocab.get_path()) as f:
            num_labels = max(list(eval(f.read()).values())) + 1  # 0-based index
            self.vocab_size.set(num_labels)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 4})
