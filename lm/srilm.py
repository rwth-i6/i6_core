__all__ = [
    "CountNgramsJob",
    "OptimizeKNDiscountsJob",
    "ComputeNgramLmJob",
    "ComputeNgramLmPerplexityJob",
    "ComputeBestMixJob",
    "InterpolateNgramLmJob",
    "PruneLMWithHelperLMJob",
]

import os
import shutil

from enum import Enum
from typing import Dict, List, Optional

from sisyphus import tk, Job, Task

from i6_core.util import create_executable, relink, get_ngram_count_exe, get_ngram_exe, get_compute_best_mix_exe


class CountNgramsJob(Job):
    """
    count ngrams with SRILM
    """

    def __init__(
        self,
        ngram_order: int,
        data: tk.Path,
        count_args: Optional[List[str]] = None,
        count_exe: Optional[tk.Path] = None,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param data: Input data to be read as textfile
        :param count_args: Extra arguments for the execution call e.g. ['-unk']
        :param count_exe: Path to srilm ngram-count executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)

        Example options/parameters for count_args:
        -unk
        """
        self.ngram_order = ngram_order
        self.data = data
        self.count_args = count_args if count_args is not None else ["-unk"]
        self.count_exe = get_ngram_count_exe(count_exe)

        self.rqmt = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

        self.out_counts = self.output_path("counts", cached=True)

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.count_exe} \\\n",
            f"  -text {self.data.get_path()} \\\n",
            f"  -order {self.ngram_order} \\\n",
            f"  -write counts \\\n",
            f"  {' '.join(self.count_args)} -memuse\n",
        ]

        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created bash script and relinks outputs from work folder to output folder"""
        self.sh("./run.sh")
        relink("counts", self.out_counts.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)


class OptimizeKNDiscountsJob(Job):
    """
    Uses SRILM to optimize Kneser-Ney discounts for a given dataset
    """

    def __init__(
        self,
        ngram_order: int,
        data: tk.Path,
        vocab: tk.Path,
        num_discounts: int,
        count_file: tk.Path,
        count_exe: Optional[tk.Path] = None,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param data: Held-out dataset to optimize discounts on
        :param vocab: Vocabulary file
        :param num_discounts: Number of discounts to optimize
        :param count_file: File to read counts from
        :param count_exe: Path to srilm ngram-count executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """
        self.ngram_order = ngram_order
        self.data = data
        self.vocab = vocab
        self.num_discounts = num_discounts
        self.count_file = count_file
        self.count_exe = get_ngram_count_exe(count_exe)

        self.rqmt = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

        self.out_multi_kn_file = self.output_path("multi_kn_file")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.count_exe} \\\n",
            f"  -order {self.ngram_order} \\\n",
            f"  -optimize-discounts {self.data.get_path()} \\\n",
            f"  -multi-kn-file kn-file.txt \\\n",
            f"  -vocab {self.vocab.get_path()} \\\n",
            f"  -num-discounts {self.num_discounts} \\\n",
            f"  -read {self.count_file.get_path()}\n",
        ]
        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created bash script and relinks outputs from work folder to output folder"""
        self.sh("./run.sh")
        relink("kn-file.txt", self.out_multi_kn_file.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)


class ComputeNgramLmJob(Job):
    """
    Generate count based LM with SRILM
    """

    class DataMode(Enum):
        TEXT = 1
        COUNT = 2

    def __init__(
        self,
        ngram_order: int,
        data: tk.Path,
        data_mode: DataMode,
        vocab: Optional[tk.Path] = None,
        ngram_args: Optional[List[str]] = None,
        count_exe: Optional[tk.Path] = None,
        multi_kn_file: Optional[tk.Path] = None,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param data: Either text file or counts file to read from, set data mode accordingly
        :param data_mode: Defines whether input format is text based or count based
        :param vocab: Vocabulary file
        :param ngram_args: Extra arguments for the execution call e.g. ['-kndiscount']
        :param count_exe: Path to srilm ngram-count exe
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)

        Example options for ngram_args:
        -kndiscount -interpolate -debug <int> -addsmooth <int>
        """
        self.ngram_order = ngram_order
        self.data = data
        self.data_mode = data_mode
        self.vocab = vocab
        if ngram_order == 1 and ngram_args is None:
            ngram_args = "-debug 0 -addsmooth 0"
        self.ngram_args = ngram_args if ngram_args is not None else []
        self.multi_kn_file = multi_kn_file

        self.count_exe = get_ngram_count_exe(count_exe)

        self.rqmt_run = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }
        self.fs_rqmt = fs_rqmt

        self.out_vocab = self.output_path("vocab", cached=True)
        self.out_ngram_lm = self.output_path("ngram.lm.gz", cached=True)

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt_run)
        yield Task(
            "compress",
            resume="compress",
            rqmt={"mem": 2, "time": 2, "cpu": 1, "fs": self.fs_rqmt},
        )

    def create_files(self):
        """creates bash script for lm creation and compression that will be executed in the run Task"""
        vocab_str = (
            f"  -vocab {self.vocab.get_cached_path()} \\\n" if self.vocab is not None else "  -write-vocab vocab \\\n"
        )
        multi_kn_str = (
            f"  -multi-kn-file {self.multi_kn_file.get_path()} \\\n" if self.multi_kn_file is not None else ""
        )
        if self.data_mode == ComputeNgramLmJob.DataMode.TEXT:
            data_str = "-text"
        elif self.data_mode == ComputeNgramLmJob.DataMode.COUNT:
            data_str = "-read"
        else:
            raise NotImplementedError

        cmd = [
            f"{self.count_exe.get_path()} \\\n",
            f"  {data_str} {self.data.get_cached_path()} \\\n",
            f"{vocab_str}",
            f"  -order {self.ngram_order} \\\n",
            f"  -lm ngram.lm \\\n",
            f"{multi_kn_str}",
            f"  {' '.join(self.ngram_args)} -unk -memuse\n",
        ]
        create_executable("run.sh", cmd)
        create_executable("compress.sh", [f"gzip -c -9 ngram.lm > ngram.lm.gz\n"])

    def run(self):
        """executes the previously created lm script and relinks the vocabulary from work folder to output folder"""
        self.sh("./run.sh")
        if self.vocab is None:
            relink("vocab", self.out_vocab.get_path())
        else:
            relink(self.vocab.get_path(), self.out_vocab.get_path())

    def compress(self):
        """executes the previously created compression script and relinks the lm from work folder to output folder"""
        self.sh("./compress.sh")
        relink("ngram.lm.gz", self.out_ngram_lm.get_path())
        if os.path.exists("ngram.lm") and os.path.exists("ngram.lm.gz"):
            os.remove("ngram.lm")

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)


class ComputeNgramLmPerplexityJob(Job):
    """Calculate the Perplexity of an Ngram LM via SRILM"""

    def __init__(
        self,
        ngram_order: int,
        lm: tk.Path,
        eval_data: tk.Path,
        vocab: Optional[tk.Path] = None,
        set_unknown_flag: bool = True,
        ppl_args: Optional[str] = None,
        ngram_exe: Optional[tk.Path] = None,
        mem_rqmt: int = 16,
        time_rqmt: float = 12,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "10G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param lm: LM to evaluate
        :param vocab: Vocabulary file
        :param eval_data: Data to calculate PPL on
        :param ppl_args: Extra arguments for the execution call e.g. '-debug 2'
        :param ngram_exe: Path to srilm ngram exe
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """

        self.ngram_order = ngram_order
        self.lm = lm
        self.vocab = vocab
        self.eval_data = eval_data
        self.set_unknown_flag = set_unknown_flag
        self.ppl_args = ppl_args if ppl_args is not None else ""

        self.ngram_exe = get_ngram_exe(ngram_exe)

        self.rqmt = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

        self.out_ppl_log = self.output_path("perplexity.log", cached=True)
        self.out_ppl_score = self.output_var("perplexity.score")
        self.out_num_sentences = self.output_var("num_sentences")
        self.out_num_words = self.output_var("num_words")
        self.out_num_oovs = self.output_var("num_oovs")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("get_ppl", mini_task=True)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.ngram_exe.get_path()} \\\n",
            f"  -order {self.ngram_order} \\\n",
        ]

        if self.set_unknown_flag:
            cmd += [f"  -unk \\\n"]

        if self.vocab is not None:
            cmd += [f"  -vocab {self.vocab.get_cached_path()} \\\n"]

        cmd += [
            f"  -ppl {self.eval_data.get_cached_path()} \\\n",
            f"  -lm {self.lm.get_cached_path()} \\\n",
            f"  {self.ppl_args} &> ppl.log\n",
        ]

        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created script and relinks the log file from work folder to output folder"""
        self.sh("./run.sh")
        relink("ppl.log", self.out_ppl_log.get_path())

    def get_ppl(self):
        """extracts various outputs from the ppl.log file"""
        with open(self.out_ppl_log.get_cached_path(), "rt") as f:
            lines = f.readlines()[-2:]
            for line in lines:
                line = line.split(" ")
                for idx, ln in enumerate(line):
                    if ln == "sentences,":
                        self.out_num_sentences.set(int(line[idx - 1]))
                    if ln == "words,":
                        self.out_num_words.set(int(line[idx - 1]))
                    if ln == "OOVs":
                        self.out_num_oovs.set(int(line[idx - 1]))
                    if ln == "ppl=":
                        self.out_ppl_score.set(float(line[idx + 1]))

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)


class ComputeBestMixJob(Job):
    """Compute the best mixture weights from given PPL logs"""

    def __init__(self, ppl_log: List[tk.Path], compute_best_mix_exe: Optional[tk.Path] = None):
        """

        :param ppl_log: List of PPL Logs to compute the weights from
        :param compute_best_mix_exe: Path to srilm compute_best_mix executable (not hashed)
        """
        self.ppl_log = ppl_log
        self.compute_best_mix_exe = get_compute_best_mix_exe(compute_best_mix_exe)

        self.out_lambdas = [self.output_var(f"lambdas{i}") for i, p in enumerate(ppl_log)]
        self.out_cbm_file = self.output_path("cbm.log")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _get_cmd(self) -> str:
        """creates command string for the bash call"""
        cmd = self.compute_best_mix_exe.get_path()

        cmd += " "

        ppl_log = [x.get_path() for x in self.ppl_log]

        cmd += " ".join(ppl_log)

        cmd += " &> cbm.log"

        return cmd

    def run(self):
        """Call the srilm script and extracts the different weights from the log, then relinks log to output folder"""
        cmd = self._get_cmd()
        self.sh(cmd)

        lines = open("cbm.log", "rt").readlines()
        lbds = lines[-1].split("(")[1].replace(")", "")
        lbds = lbds.split()

        for i, v in enumerate(lbds):
            self.out_lambdas[i].set(float(v))

        relink("cbm.log", self.out_cbm_file.get_path())

    @classmethod
    def hash(cls, parsed_args):
        """delete the executable from the hashing"""
        del parsed_args["compute_best_mix_exe"]
        return super().hash(parsed_args)


class InterpolateNgramLmJob(Job):
    """Uses SRILM to interpolate different LMs with previously calculated weights"""

    def __init__(
        self,
        ngram_lms: List[tk.Path],
        lambdas: List[tk.Variable],  # List[float]
        ngram_order: int,
        interpolation_args: Optional[Dict] = None,
        ngram_exe: Optional[tk.Path] = None,
        cpu_rqmt: int = 1,
        mem_rqmt: int = 32,
        time_rqmt: int = 4,
        fs_rqmt: str = "50G",
    ):
        """

        :param ngram_lms: List of language models to interpolate
        :param lambdas: Weights of different language models, has to be same order as LMs
        :param ngram_order: Maximum n gram order
        :param interpolation_args: Additional arguments for interpolation
        :param ngram_exe: Path to srilm ngram executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """
        self.ngram_lms = ngram_lms
        self.lambdas = lambdas
        self.ngram_order = ngram_order
        self.interpolation_args = interpolation_args if interpolation_args is not None else {}
        self.ngram_exe = get_ngram_exe(ngram_exe)

        assert len(ngram_lms) >= 2
        assert len(ngram_lms) == len(lambdas), (
            "ngram list len:",
            len(ngram_lms),
            ngram_lms,
            "\nlambda weight list len:",
            len(lambdas),
            lambdas,
        )

        self.rqmt = {
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

        self.out_interpolated_lm = self.output_path("interpolated.txt.gz")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _get_cmd(self) -> str:
        """creates command string for the bash call"""
        cmd = self.ngram_exe.get_path()
        cmd += f" -order {self.ngram_order} -unk"

        for i, lm in enumerate(self.ngram_lms):
            if i == 0:
                c = f" -lm {lm.get_path()}"
            elif i == 1:
                c = f" -mix-lm {lm.get_path()}"
            else:
                c = f" -mix-lm{i} {lm.get_path()}"
            cmd += c

        for i, lmbd in enumerate(self.lambdas):
            if i == 0:
                c = f" -lambda {lmbd.get()}"
            elif i == 1:
                c = f""
            else:
                c = f" -mix-lambda{i} {lmbd.get()}"
            cmd += c
        cmd += " -write-lm interpolated.lm.gz"

        return cmd

    def run(self):
        """delete the executable from the hashing"""
        cmd = self._get_cmd()
        self.sh(cmd)
        shutil.move("interpolated.lm.gz", self.out_interpolated_lm.get_path())

    @classmethod
    def hash(cls, parsed_args):
        """delete the queue requirements from the hashing"""
        del parsed_args["cpu_rqmt"]
        del parsed_args["mem_rqmt"]
        del parsed_args["time_rqmt"]
        del parsed_args["fs_rqmt"]
        return super().hash(parsed_args)


class PruneLMWithHelperLMJob(Job):
    """
    Job that prunes the given LM with the help of a helper LM
    """

    def __init__(
        self,
        ngram_order: int,
        lm: tk.Path,
        prune_thresh: float,
        helper_lm: tk.Path,
        ngram_exe: Optional[tk.Path] = None,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """

        :param ngram_order: Maximum n gram order
        :param lm: LM to be pruned
        :param prune_thresh: Pruning threshold
        :param helper_lm: helper/'Katz' LM to prune the other LM with
        :param ngram_exe: Path to srilm ngram-count executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """

        self.ngram_order = ngram_order
        self.lm = lm
        self.prune_thresh = prune_thresh
        self.helper_lm = helper_lm
        self.ngram_exe = get_ngram_exe(ngram_exe)

        self.out_lm = self.output_path("pruned_lm.gz")
        self.rqmt_run = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt_run)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.ngram_exe} \\\n",
            f"  -order {self.ngram_order} \\\n",
            f"  -renorm -unk \\\n",
            f"  -lm {self.lm.get_path()} \\\n",
            f"  -write-lm pruned.lm.gz \\\n",
            f"  -prune {self.prune_thresh} \\\n",
            f"  -prune-history-lm {self.helper_lm.get_path()} \\\n",
            f"  -memuse \n",
        ]
        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created script and relinks the lm from work folder to output folder"""
        self.sh("./run.sh")
        relink("pruned.lm.gz", self.out_lm.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)
