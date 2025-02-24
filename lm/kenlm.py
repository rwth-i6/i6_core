import shutil

from sisyphus import Job, Task, tk, gs

from typing import Union, List, Optional, Sequence
import os
import tempfile
import subprocess as sp

from i6_core.util import uopen


class CompileKenLMJob(Job):
    """
    Compile KenLM and store a folder containing the binaries.

    Please make sure the needed libraries (e.g. boost, zlib) are on your system or image.
    On Ubuntu: build-essential libeigen3-dev libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    """

    def __init__(self, *, repository: tk.Path):
        """

        :param repository: e.g. CloneGitRepositoryJob output for https://github.com/kpu/kenlm
        """
        self.repository = repository
        self.out_binaries = self.output_path("kenlm_binary_folder")
        self.rqmt = {"time": 0.5, "mem": 4, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as td:
            repo = os.path.join(td, "repo")
            shutil.copytree(self.repository.get_path(), repo)
            build_dir = os.path.join(repo, "build")
            os.mkdir(build_dir)
            sp.check_call(["cmake", ".."], cwd=build_dir)
            sp.check_call(["make", "-j", str(self.rqmt["cpu"])], cwd=build_dir)
            shutil.copytree(os.path.join(build_dir, "bin"), self.out_binaries.get_path())


class KenLMplzJob(Job):
    """
    Run the lmplz command of the KenLM toolkit to create a gzip compressed ARPA-LM file
    """

    __sis_hash_exclude__ = {"discount_fallback": None}

    def __init__(
        self,
        *,
        text: Union[tk.Path, List[tk.Path]],
        order: int,
        interpolate_unigrams: bool,
        pruning: Optional[List[int]],
        vocabulary: Optional[tk.Path],
        discount_fallback: Optional[Sequence[Union[float, int]]] = None,
        kenlm_binary_folder: tk.Path,
        mem: float = 4.0,
        time: float = 1.0,
    ):
        """

        :param text: training text data
        :param order: "N"-order of the "N"-gram LM
        :param interpolate_unigrams: Set True for KenLM default, and False for SRILM-compatibility.
            Having this as False will increase the share of the unknown probability
        :param pruning: absolute pruning threshold for each order,
            e.g. to remove 3-gram and 4-gram singletons in a 4th order model use [0, 0, 1, 1]
        :param vocabulary: a "single word per line" file to determine valid words,
            everything else will be treated as unknown
        :param discount_fallback: This option falls back to user-specified discounts
            when the closed-form estimate fails.
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        :param mem: memory rqmt, needs adjustment for large training corpora
        :param time: time rqmt, might adjustment for very large training corpora and slow machines
        """
        self.text = text
        self.order = order
        self.interpolate_unigrams = interpolate_unigrams
        self.pruning = pruning
        self.vocabulary = vocabulary
        self.discount_fallback = discount_fallback
        self.kenlm_binary_folder = kenlm_binary_folder

        self.out_lm = self.output_path("lm.gz")

        self.rqmt = {"cpu": 1, "mem": mem, "time": time}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            lmplz_command = [
                os.path.join(self.kenlm_binary_folder.get_path(), "lmplz"),
                "-o",
                str(self.order),
                "--interpolate_unigrams",
                str(self.interpolate_unigrams),
                "-S",
                "%dG" % int(self.rqmt["mem"]),
                "-T",
                tmp,
            ]
            if self.pruning is not None:
                lmplz_command += ["--prune"] + [str(p) for p in self.pruning]
            if self.vocabulary is not None:
                lmplz_command += ["--limit_vocab_file", self.vocabulary.get_path()]
            if self.discount_fallback is not None:
                lmplz_command += ["--discount_fallback"] + [str(d) for d in self.discount_fallback]

            zcat_command = ["zcat", "-f"] + [text.get_path() for text in self.text]
            with uopen(self.out_lm, "wb") as lm_file:
                p1 = sp.Popen(zcat_command, stdout=sp.PIPE)
                p2 = sp.Popen(lmplz_command, stdin=p1.stdout, stdout=sp.PIPE)
                sp.check_call("gzip", stdin=p2.stdout, stdout=lm_file)
                p2.wait()
                if p2.returncode:
                    raise sp.CalledProcessError(p2.returncode, cmd=lmplz_command)

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mem"]
        del parsed_args["time"]
        return super().hash(parsed_args)


class CreateBinaryLMJob(Job):
    """
    Run the build_binary command of the KenLM toolkit to create a binary LM from an given ARPA LM
    """

    def __init__(
        self,
        *,
        arpa_lm: tk.Path,
        kenlm_binary_folder: tk.Path,
    ):
        """
        :param arpa_lm: any ARPA format LM
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        """
        self.arpa_lm = arpa_lm
        self.kenlm_binary_folder = kenlm_binary_folder

        self.out_lm = self.output_path("lm.bin")

        self.rqmt = {"cpu": 1, "mem": 8.0, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        build_binary = os.path.join(self.kenlm_binary_folder.get_path(), "build_binary")
        sp.check_call([build_binary, self.arpa_lm.get_path(), self.out_lm.get_path()])
