__all__ = [
    "CountNgramsJob",
    "DiscountNgramsJob",
    "ComputeNgramLmJob",
    "ComputeNgramLmPerplexityJob",
    "ComputeBestMixJob",
    "InterpolateNgramLmJob",
    "PruneLMWithHelperLMJob",
]

import os
import shutil
import subprocess

from enum import Enum
from typing import Dict, List, Optional

from sisyphus import tk, Job, Task

from i6_core.util import create_executable, relink


class CountNgramsJob(Job):
    """
    Count ngrams with SRILM
    """

    def __init__(
        self,
        ngram_order: int,
        data: tk.Path,
        count_exe: tk.Path,
        *,
        extra_count_args: Optional[List[str]] = None,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param data: Input data to be read as textfile
        :param extra_count_args: Extra arguments for the execution call e.g. ['-unk']
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
        self.count_args = extra_count_args if extra_count_args is not None else ["-unk"]
        self.count_exe = count_exe

        self.out_counts = self.output_path("counts", cached=True)

        self.rqmt = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.count_exe.get_path()} \\\n",
            f"  -text {self.data.get_path()} \\\n",
            f"  -order {self.ngram_order} \\\n",
            f"  -write counts \\\n",
            f"  {' '.join(self.count_args)} -memuse\n",
        ]

        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created bash script and relinks outputs from work folder to output folder"""
        subprocess.check_call("./run.sh")
        relink("counts", self.out_counts.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)


class DiscountNgramsJob(Job):
    """
    Create a file with the discounted ngrams with SRILM
    """

    __sis_hash_exclude__ = {"data_for_optimization": None}

    def __init__(
        self,
        ngram_order: int,
        counts: tk.Path,
        count_exe: tk.Path,
        *,
        vocab: Optional[tk.Path] = None,
        data_for_optimization: Optional[tk.Path] = None,
        extra_discount_args: Optional[List[str]] = None,
        use_modified_srilm: bool = False,
        cpu_rqmt: int = 1,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
    ):
        """
        :param ngram_order: order of the ngram counts, typically 3 or 4.
        :param counts: file with the ngram counts, see :class:`CountNgramsJob.out_counts`.
        :param count_exe: path to the binary.
        :param vocab: vocabulary file for the discounting.
        :param data_for_optimization: the discounting will be optimized on this dataset.
        :param extra_discount_args: additional arguments for the discounting step.
        :param use_modified_srilm: Use the i6 modified SRILM version by Sundermeyer.
                                   The SRILM binary ngram-count was modified.
                                   This version is currently only available internally.
        :param cpu_rqmt: CPU requirements.
        :param mem_rqmt: memory requirements.
        :param time_rqmt: time requirements.
        """
        self.ngram_order = ngram_order
        self.counts = counts
        self.vocab = vocab
        self.data_for_opt = data_for_optimization
        self.discount_args = extra_discount_args or []
        self.use_modified_srilm = use_modified_srilm

        self.count_exe = count_exe

        self.out_discounts = self.output_path("discounts", cached=True)

        self.rqmt_run = {
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt_run)

    def create_files(self):
        """creates bash script to compute discounts"""
        cmd = [
            f"{self.count_exe.get_path()} \\\n",
            f"  -order {self.ngram_order} \\\n",
        ]
        if self.vocab is not None:
            cmd.append(f"  -vocab {self.vocab.get_cached_path()} \\\n")
        cmd += ["  -kn discounts\\\n"] if not self.use_modified_srilm else [f"  -multi-kn-file discounts \\\n"]
        if self.data_for_opt is not None:
            cmd.append(f"  -optimize-discounts {self.data_for_opt.get_cached_path()} \\\n")
        cmd += [
            f"  -read {self.counts.get_cached_path()} \\\n",
            f"  {' '.join(self.discount_args)} -memuse\n",
        ]

        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created bash script and relinks outputs from work folder to output folder"""
        subprocess.check_call("./run.sh")
        relink("discounts", self.out_discounts.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        return super().hash(kwargs)


class ComputeNgramLmJob(Job):
    """
    Generate count based LM with SRILM
    """

    __sis_hash_exclude__ = {
        "discounts": None,
        "use_modified_srilm": False,
    }

    class DataMode(Enum):
        TEXT = 1
        COUNT = 2

    def __init__(
        self,
        ngram_order: int,
        data: tk.Path,
        data_mode: DataMode,
        count_exe: tk.Path,
        *,
        vocab: Optional[tk.Path] = None,
        discounts: Optional[tk.Path] = None,
        extra_ngram_args: Optional[List[str]] = None,
        use_modified_srilm: bool = False,
        mem_rqmt: int = 48,
        time_rqmt: float = 24,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param data: Either text file or counts file to read from, set data mode accordingly
                     the counts file can come from the `CountNgramsJob.out_counts`
        :param data_mode: Defines whether input format is text based or count based
        :param vocab: Vocabulary file, one word per line
        :param discounts: Discounting file from :class:`DiscountNgramsJob`.
        :param extra_ngram_args: Extra arguments for the execution call e.g. ['-kndiscount']
        :param use_modified_srilm: Use the i6 modified SRILM version by Sundermeyer.
                                   The SRILM binary ngram-count was modified.
                                   This version is currently only available internally.
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
        self.discounts = discounts
        self.ngram_args = extra_ngram_args if extra_ngram_args is not None else []
        self.use_modified_srilm = use_modified_srilm

        self.count_exe = count_exe

        self.out_vocab = self.output_path("vocab", cached=True)
        self.out_ngram_lm = self.output_path("ngram.lm.gz", cached=True)

        self.rqmt_run = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }
        self.fs_rqmt = fs_rqmt

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt_run)
        yield Task(
            "compress",
            rqmt={"mem": 2, "time": 2, "cpu": 1, "fs": self.fs_rqmt},
        )

    def create_files(self):
        """creates bash script for lm creation and compression that will be executed in the run Task"""
        vocab_str = (
            f"  -vocab {self.vocab.get_cached_path()} \\\n" if self.vocab is not None else "  -write-vocab vocab \\\n"
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
        ]

        if self.discounts is not None:
            if not self.use_modified_srilm:
                cmd.append(f"  -kn {self.discounts.get_path()}\\\n")
            else:
                cmd.append(f"  -multi-kn-file {self.discounts.get_path()}\\\n")

        cmd += [
            f"  {' '.join(self.ngram_args)} -unk -memuse\n",
        ]
        create_executable("run.sh", cmd)
        create_executable("compress.sh", [f"gzip -c -9 ngram.lm > ngram.lm.gz\n"])

    def run(self):
        """executes the previously created lm script and relinks the vocabulary from work folder to output folder"""
        subprocess.check_call("./run.sh")
        if self.vocab is None:
            relink("vocab", self.out_vocab.get_path())
        else:
            shutil.copy(self.vocab.get_path(), self.out_vocab.get_path())

    def compress(self):
        """executes the previously created compression script and relinks the lm from work folder to output folder"""
        subprocess.check_call("./compress.sh")
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
    """
    Calculate the Perplexity of a Ngram LM via SRILM
    """

    def __init__(
        self,
        ngram_order: int,
        lm: tk.Path,
        eval_data: tk.Path,
        ngram_exe: tk.Path,
        *,
        vocab: Optional[tk.Path] = None,
        set_unknown_flag: bool = True,
        extra_ppl_args: Optional[str] = None,
        mem_rqmt: int = 16,
        time_rqmt: float = 12,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "10G",
    ):
        """
        :param ngram_order: Maximum n gram order
        :param lm: LM to evaluate
        :param eval_data: Data to calculate PPL on
        :param vocab: Vocabulary file
        :param set_unknown_flag: sets unknown lemma
        :param extra_ppl_args: Extra arguments for the execution call e.g. '-debug 2'
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
        self.ppl_args = extra_ppl_args if extra_ppl_args is not None else ""
        self.ngram_exe = ngram_exe

        self.out_ppl_log = self.output_path("perplexity.log", cached=True)
        self.out_ppl_score = self.output_var("perplexity.score")
        self.out_num_sentences = self.output_var("num_sentences")
        self.out_num_words = self.output_var("num_words")
        self.out_num_oovs = self.output_var("num_oovs")

        self.rqmt = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)
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
        subprocess.check_call("./run.sh")
        relink("ppl.log", self.out_ppl_log.get_path())

    def get_ppl(self):
        """extracts various outputs from the ppl.log file"""
        with open(self.out_ppl_log.get_path(), "rt") as f:
            lines = f.readlines()[-2:]
            for line in lines:
                line = line.split(" ")
                for idx, ln in enumerate(line):
                    if ln == "sentences,":
                        self.out_num_sentences.set(int(line[idx - 1]))
                    if ln == "words,":
                        self.out_num_words.set(int(float(line[idx - 1])))
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
    """
    Compute the best mixture weights for a combination of count LMs based on the given PPL logs
    """

    def __init__(self, ppl_logs: List[tk.Path], compute_best_mix_exe: tk.Path):
        """

        :param ppl_logs: List of PPL Logs to compute the weights from
        :param compute_best_mix_exe: Path to srilm compute_best_mix executable
        """
        self.ppl_logs = ppl_logs
        self.compute_best_mix_exe = compute_best_mix_exe

        self.out_weights = [self.output_var(f"weights{i}") for i, p in enumerate(ppl_logs)]
        self.out_cbm_file = self.output_path("cbm.log")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _create_cmd(self) -> List[str]:
        """:return: the command"""
        cmd = [self.compute_best_mix_exe.get_path()]

        ppl_log = [x.get_path() for x in self.ppl_logs]

        cmd += ppl_log

        return cmd

    def run(self):
        """Call the srilm script and extracts the different weights from the log, then relinks log to output folder"""
        cmd = self._create_cmd()
        subprocess.check_call(cmd, stdout=open("cbm.log", "wb"), stderr=subprocess.STDOUT)

        lines = open("cbm.log", "rt").readlines()
        lbds = lines[-2].split("(")[1].split(")")[0]
        lbds = lbds.split()

        for i, v in enumerate(lbds):
            self.out_weights[i].set(float(v))

        relink("cbm.log", self.out_cbm_file.get_path())


class InterpolateNgramLmJob(Job):
    """
    Uses SRILM to interpolate different LMs with previously calculated weights
    """

    def __init__(
        self,
        ngram_lms: List[tk.Path],
        weights: List[tk.Variable],
        ngram_order: int,
        ngram_exe: tk.Path,
        *,
        extra_interpolation_args: Optional[Dict] = None,
        cpu_rqmt: int = 1,
        mem_rqmt: int = 32,
        time_rqmt: int = 4,
        fs_rqmt: str = "50G",
    ):
        """

        :param ngram_lms: List of language models to interpolate, format: ARPA, compressed ARPA
        :param weights: Weights of different language models, has to be same order as ngram_lms
        :param ngram_order: Maximum n gram order
        :param extra_interpolation_args: Additional arguments for interpolation
        :param ngram_exe: Path to srilm ngram executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """
        self.ngram_lms = ngram_lms
        self.weights = weights
        self.ngram_order = ngram_order
        self.interpolation_args = extra_interpolation_args if extra_interpolation_args is not None else {}
        self.ngram_exe = ngram_exe

        assert len(ngram_lms) >= 2
        assert len(ngram_lms) == len(weights), (
            "ngram list len:",
            len(ngram_lms),
            ngram_lms,
            "\nlambda weight list len:",
            len(weights),
            weights,
        )

        self.out_interpolated_lm = self.output_path("interpolated.txt.gz")

        self.rqmt = {
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _create_cmd(self) -> List[str]:
        """:return: the command"""
        cmd = [self.ngram_exe.get_path()]
        cmd += ["-order", str(self.ngram_order), "-unk"]

        for i, lm in enumerate(self.ngram_lms):
            if i == 0:
                c = ["-lm", lm.get_path()]
            elif i == 1:
                c = ["-mix-lm", lm.get_path()]
            else:
                c = [f"-mix-lm{i}", lm.get_path()]
            cmd += c

        for i, lmbd in enumerate(self.weights):
            if i == 0:
                c = ["-lambda", str(lmbd.get())]
            elif i == 1:
                c = []
            else:
                c = [f"-mix-lambda{i}", str(lmbd.get())]
            cmd += c
        cmd += ["-write-lm", "interpolated.lm.gz"]

        return cmd

    def run(self):
        """run the command"""
        cmd = self._create_cmd()
        subprocess.check_call(cmd)
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
        ngram_exe: tk.Path,
        *,
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
        self.ngram_exe = ngram_exe

        self.out_lm = self.output_path("pruned_lm.gz")

        self.rqmt_run = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt_run)

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
        subprocess.check_call("./run.sh")
        relink("pruned.lm.gz", self.out_lm.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)
