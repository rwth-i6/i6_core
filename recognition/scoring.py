__all__ = [
    "AnalogJob",
    "ScliteJob",
    "Hub5ScoreJob",
    "QuaeroScorerJob",
    "KaldiScorerJob",
    "IntersectStmCtm",
]

import os
import shutil
import subprocess as sp
import tempfile
import collections
import re
from typing import List, Optional, Dict, Tuple

from sisyphus import *
from i6_core.lib.corpus import *
from i6_core.util import uopen

Path = setup_path(__package__)


class AnalogJob(Job):
    def __init__(self, configs, merge=True):
        self.set_vis_name("Analog")

        self.merge = merge
        self.configs = configs
        if type(configs) == dict:
            self.configs = list(configs.values())
        self.out_report = self.output_path("report.analog")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        analog_path = os.path.join(gs.RASR_ROOT, "src/Tools/Analog/analog")
        with open(self.out_report.get_path(), "w") as out:
            sp.check_call(
                [analog_path] + (["-m"] if self.merge else []) + [tk.uncached_path(c) for c in self.configs],
                stdout=out,
            )


class ScliteJob(Job):
    """
    Run the Sclite scorer from the SCTK toolkit

    Outputs:
        - out_report_dir: contains the report files with detailed scoring information
        - out_*: the job also outputs many variables, please look in the init code for a list
    """

    __sis_hash_exclude__ = {"sctk_binary_path": None, "precision_ndigit": 1}

    def __init__(
        self,
        ref: tk.Path,
        hyp: tk.Path,
        cer: bool = False,
        sort_files: bool = False,
        additional_args: Optional[List[str]] = None,
        sctk_binary_path: Optional[tk.Path] = None,
        precision_ndigit: Optional[int] = 1,
    ):
        """
        :param ref: reference stm text file
        :param hyp: hypothesis ctm text file
        :param cer: compute character error rate
        :param sort_files: sort ctm and stm before scoring
        :param additional_args: additional command line arguments passed to the Sclite binary call
        :param sctk_binary_path: set an explicit binary path.
        :param precision_ndigit: number of digits after decimal point for the precision
            of the percentages in the output variables.
            If None, no rounding is done.
            In sclite, the precision was always one digit after the decimal point
            (https://github.com/usnistgov/SCTK/blob/f48376a203ab17f/src/sclite/sc_dtl.c#L343),
            thus we recalculate the percentages here.
        """
        self.set_vis_name("Sclite - %s" % ("CER" if cer else "WER"))

        self.ref = ref
        self.hyp = hyp
        self.cer = cer
        self.sort_files = sort_files
        self.additional_args = additional_args
        self.sctk_binary_path = sctk_binary_path
        self.precision_ndigit = precision_ndigit

        self.out_report_dir = self.output_path("reports", True)

        self.out_wer = self.output_var("wer")
        self.out_num_errors = self.output_var("num_errors")
        self.out_percent_correct = self.output_var("percent_correct")
        self.out_num_correct = self.output_var("num_correct")
        self.out_percent_substitution = self.output_var("percent_substitution")
        self.out_num_substitution = self.output_var("num_substitution")
        self.out_percent_deletions = self.output_var("percent_deletions")
        self.out_num_deletions = self.output_var("num_deletions")
        self.out_percent_insertions = self.output_var("percent_insertions")
        self.out_num_insertions = self.output_var("num_insertions")
        self.out_percent_word_accuracy = self.output_var("percent_word_accuracy")
        self.out_ref_words = self.output_var("ref_words")
        self.out_hyp_words = self.output_var("hyp_words")
        self.out_aligned_words = self.output_var("aligned_words")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self, output_to_report_dir=True):
        if self.sort_files:
            sort_stm_args = ["sort", "-k1,1", "-k4,4n", self.ref.get_path()]
            (fd_stm, tmp_stm_file) = tempfile.mkstemp(suffix=".stm")
            res = sp.run(sort_stm_args, stdout=sp.PIPE)
            os.write(fd_stm, res.stdout)
            os.close(fd_stm)

            sort_ctm_args = ["sort", "-k1,1", "-k3,3n", self.hyp.get_path()]
            (fd_ctm, tmp_ctm_file) = tempfile.mkstemp(suffix=".ctm")
            res = sp.run(sort_ctm_args, stdout=sp.PIPE)
            os.write(fd_ctm, res.stdout)
            os.close(fd_ctm)

        if self.sctk_binary_path:
            sclite_path = os.path.join(self.sctk_binary_path.get_path(), "sclite")
        else:
            sclite_path = os.path.join(gs.SCTK_PATH, "sclite") if hasattr(gs, "SCTK_PATH") else "sclite"
        output_dir = self.out_report_dir.get_path() if output_to_report_dir else "."
        stm_file = tmp_stm_file if self.sort_files else self.ref.get_path()
        ctm_file = tmp_ctm_file if self.sort_files else self.hyp.get_path()

        args = [
            sclite_path,
            "-r",
            stm_file,
            "stm",
            "-h",
            ctm_file,
            "ctm",
            "-o",
            "all",
            "-o",
            "dtl",
            "-o",
            "lur",
            "-n",
            "sclite",
            "-O",
            output_dir,
        ]
        if self.cer:
            args.append("-c")
        if self.additional_args is not None:
            args += self.additional_args

        sp.check_call(args)

        if output_to_report_dir:  # run as real job
            with open(f"{output_dir}/sclite.dtl", "rt", errors="ignore") as f:
                # Example:
                """
                Percent Total Error       =    5.3%   (2709)
                ...
                Percent Word Accuracy     =   94.7%
                ...
                Ref. words                =           (50948)
                """

                # key -> percentage, absolute
                output_variables: Dict[str, Tuple[Optional[tk.Variable], Optional[tk.Variable]]] = {
                    "Percent Total Error": (self.out_wer, self.out_num_errors),
                    "Percent Correct": (self.out_percent_correct, self.out_num_correct),
                    "Percent Substitution": (self.out_percent_substitution, self.out_num_substitution),
                    "Percent Deletions": (self.out_percent_deletions, self.out_num_deletions),
                    "Percent Insertions": (self.out_percent_insertions, self.out_num_insertions),
                    "Percent Word Accuracy": (self.out_percent_word_accuracy, None),
                    "Ref. words": (None, self.out_ref_words),
                    "Hyp. words": (None, self.out_hyp_words),
                    "Aligned words": (None, self.out_aligned_words),
                }

                outputs_absolute: Dict[str, int] = {}
                for line in f:
                    key: Optional[str] = ([key for key in output_variables if line.startswith(key)] or [None])[0]
                    if not key:
                        continue
                    pattern = rf"^{re.escape(key)}\s*=\s*((\S+)%)?\s*(\(\s*(\d+)\))?$"
                    m = re.match(pattern, line)
                    assert m, f"Could not parse line: {line!r}, does not match to pattern r'{pattern}'"
                    absolute_s = m.group(4)
                    if not absolute_s:
                        assert not output_variables[key][1], f"Expected absolute value for {key}"
                        continue
                    outputs_absolute[key] = int(absolute_s)
                    if key == "Aligned words":
                        break  # that should be the last key, can stop now

                assert "Ref. words" in outputs_absolute, "Expected absolute numbers for Ref. words"
                num_ref_words = outputs_absolute["Ref. words"]
                assert "Percent Total Error" in outputs_absolute, "Expected absolute numbers for Percent Total Error"
                outputs_absolute["Percent Word Accuracy"] = num_ref_words - outputs_absolute["Percent Total Error"]

                outputs_percentage: Dict[str, float] = {}
                for key, absolute in outputs_absolute.items():
                    if num_ref_words > 0:
                        percentage = 100.0 * absolute / num_ref_words
                    else:
                        percentage = float("nan")
                    outputs_percentage[key] = (
                        round(percentage, self.precision_ndigit) if self.precision_ndigit is not None else percentage
                    )

                for key, (percentage_var, absolute_var) in output_variables.items():
                    if percentage_var is not None:
                        assert key in outputs_percentage, f"Expected percentage value for {key}"
                        percentage_var.set(outputs_percentage[key])
                    if absolute_var is not None:
                        assert key in outputs_absolute, f"Expected absolute value for {key}"
                        absolute_var.set(outputs_absolute[key])

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(output_to_report_dir=False)
            dtl_file = "sclite.dtl"
            with open(dtl_file, "rt", errors="ignore") as f:
                for line in f:
                    if line.startswith("Percent Total Error"):
                        errors = float("".join(line.split()[5:])[1:-1])
                    if line.startswith("Ref. words"):
                        wer = 100.0 * errors / float("".join(line.split()[3:])[1:-1])
                        break
        os.chdir(old_dir)

        return wer


class Hub5ScoreJob(Job):
    __sis_hash_exclude__ = {"sctk_binary_path": None}

    def __init__(
        self,
        ref: tk.Path,
        glm: tk.Path,
        hyp: tk.Path,
        sctk_binary_path: Optional[tk.Path] = None,
    ):
        """
        :param ref: reference stm text file
        :param glm: text file containing mapping rules for scoring
        :param hyp: hypothesis ctm text file
        :param sctk_binary_path: set an explicit binary path.
        """
        self.set_vis_name("HubScore")

        self.glm = glm
        self.hyp = hyp
        self.ref = ref
        self.sctk_binary_path = sctk_binary_path

        self.out_report_dir = self.output_path("reports", True)

        self.out_wer = self.output_var("wer")
        self.out_num_errors = self.output_var("num_errors")
        self.out_percent_correct = self.output_var("percent_correct")
        self.out_num_correct = self.output_var("num_correct")
        self.out_percent_substitution = self.output_var("percent_substitution")
        self.out_num_substitution = self.output_var("num_substitution")
        self.out_percent_deletions = self.output_var("percent_deletions")
        self.out_num_deletions = self.output_var("num_deletions")
        self.out_percent_insertions = self.output_var("percent_insertions")
        self.out_num_insertions = self.output_var("num_insertions")
        self.out_percent_word_accuracy = self.output_var("percent_word_accuracy")
        self.out_ref_words = self.output_var("ref_words")
        self.out_hyp_words = self.output_var("hyp_words")
        self.out_aligned_words = self.output_var("aligned_words")

        self.out_swb_num_errors = self.output_var("swb_num_errors")
        self.out_swb_ref_words = self.output_var("swb_ref_words")
        self.out_ch_num_errors = self.output_var("ch_num_errors")
        self.out_ch_ref_words = self.output_var("ch_ref_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self, move_files=True):
        sctk_path = ""
        if self.sctk_binary_path is not None:
            sctk_path = self.sctk_binary_path.get_path()
        elif hasattr(gs, "SCTK_PATH"):
            sctk_path = gs.SCTK_PATH
        hubscr_path = os.path.join(sctk_path, "hubscr.pl")  # evaluates to just "hubscr.pl" if sctk_path is empty

        sctk_opt = ["-p", sctk_path] if sctk_path else []

        ref = self.ref
        try:
            ref = shutil.copy(ref.get_path(), ".")
        except shutil.SameFileError:
            pass

        hyp = self.hyp
        try:
            hyp = shutil.copy(hyp.get_path(), ".")
        except shutil.SameFileError:
            pass

        sp.check_call(
            [hubscr_path, "-V", "-l", "english", "-h", "hub5"] + sctk_opt + ["-g", self.glm.get_path(), "-r", ref, hyp]
        )

        if move_files:  # run as real job
            with open(f"{hyp}.filt.dtl", "rt") as f:
                for line in f:
                    s = line.split()
                    if line.startswith("Percent Total Error"):
                        self.out_wer.set(float(s[4][:-1]))
                        self.out_num_errors.set(int("".join(s[5:])[1:-1]))
                    elif line.startswith("Percent Correct"):
                        self.out_percent_correct.set(float(s[3][:-1]))
                        self.out_num_correct.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Substitution"):
                        self.out_percent_substitution.set(float(s[3][:-1]))
                        self.out_num_substitution.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Deletions"):
                        self.out_percent_deletions.set(float(s[3][:-1]))
                        self.out_num_deletions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Insertions"):
                        self.out_percent_insertions.set(float(s[3][:-1]))
                        self.out_num_insertions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Word Accuracy"):
                        self.out_percent_word_accuracy.set(float(s[4][:-1]))
                    elif line.startswith("Ref. words"):
                        self.out_ref_words.set(int(s[3][1:-1]))
                    elif line.startswith("Hyp. words"):
                        self.out_hyp_words.set(int(s[3][1:-1]))
                    elif line.startswith("Aligned words"):
                        self.out_aligned_words.set(int(s[3][1:-1]))

            with open(f"{hyp}.filt.raw", "rt") as f:
                swb_err = 0
                swb_ref = 0
                ch_err = 0
                ch_ref = 0
                for line in f:
                    s = line.split()
                    if len(s) <= 1:
                        continue
                    if s[1].startswith("sw"):
                        swb_err += int(s[10])
                        swb_ref += int(s[4])
                    elif s[1].startswith("en"):
                        ch_err += int(s[10])
                        ch_ref += int(s[4])

            self.out_swb_num_errors.set(swb_err)
            self.out_swb_ref_words.set(swb_ref)
            self.out_ch_num_errors.set(ch_err)
            self.out_ch_ref_words.set(ch_ref)

            for f in os.listdir("."):
                os.rename(f, os.path.join(self.out_report_dir.get_path(), f))

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(move_files=False)
            dtl_file = os.path.basename(self.hyp) + ".filt.dtl"
            with open(dtl_file, "rt") as f:
                for line in f:
                    if line.startswith("Percent Total Error"):
                        errors = float("".join(line.split()[5:])[1:-1])
                    if line.startswith("Ref. words"):
                        wer = 100.0 * errors / float(line.split()[3][1:-1])
                        break
        os.chdir(old_dir)

        return wer


class QuaeroScorerJob(Job):
    def __init__(self, hyp, uem, trs, glm, normalization_script, eval_script):
        self.hyp = hyp
        self.uem = uem
        self.trs = trs
        self.glm = glm

        self.normalization_script = normalization_script
        self.eval_script = eval_script

        self.out_report_dir = self.output_path("reports", True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self, move_files=True):
        hyp = self.hyp
        try:
            hyp = shutil.copy(tk.uncached_path(hyp), ".")
        except shutil.SameFileError:
            pass

        sp.check_call(
            [
                tk.uncached_path(self.eval_script),
                "-n",
                tk.uncached_path(self.normalization_script),
                "-g",
                tk.uncached_path(self.glm),
                "-u",
                tk.uncached_path(self.uem),
                "-o",
                "./quaero",
                tk.uncached_path(self.trs),
                tk.uncached_path(hyp),
            ]
        )

        if move_files:
            for f in os.listdir("."):
                os.rename(f, os.path.join(self.out_report_dir.get_path(), f))

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(move_files=False)
            dtl_file = "quaero.ci.dtl"
            with open(dtl_file, "rt") as f:
                for line in f:
                    if line.startswith("Percent Total Error"):
                        wer = float(line.split()[4][:-1])
                        break
        os.chdir(old_dir)

        return wer


class KaldiScorerJob(Job):
    """
    Applies the Kaldi compute-wer binary. Required gs.KALDI_PATH to be the path to the Kaldi bin folder.
    """

    def __init__(self, corpus_path, ctm, map, regex=None):
        """
        :param ref: Path to corpus file. This job will generate reference from it.
        :param hyp: Path to CTM file. It will be converted to Kaldi format in this Job.
        :param map: Dictionary with words to be replaced in hyp. Example: {'[NOISE]' : ''}
        :param regex: String with groups used for regex the segment names.
                      WER will be calculated for each group individually. Example: '.*(S..)(P..).*'
        """

        self.corpus_path = corpus_path
        self.ctm = ctm
        self.map = map if map else {}
        self.regex = regex

        self.out_kaldi_ref = self.output_path("ref.txt")
        self.out_kaldi_hyp = self.output_path("hyp.txt")
        self.out_report_dir = self.output_path("reports", True)
        self.out_report_path = self.output_path("reports/wer.txt")
        if regex:
            self.out_re_table = self.output_path("reports/table.txt")

    def tasks(self):
        yield Task("run", mini_task=True)
        if self.regex:
            yield Task("run_regex", mini_task=True)

    def _make_ref_regex(self):
        c = Corpus()
        c.load(tk.uncached_path(self.corpus_path))

        regex_data = {}
        regex_files = {}

        for seg in c.segments():
            name = seg.fullname()
            orth = seg.orth

            words = orth.split()
            filtered_words = [self.map.get(w, w) for w in words]
            data = " ".join(filtered_words).lower()

            res = re.match(self.regex, name)

            ids = list(res.groups())
            grouped_id = "_".join(res.groups())
            ids.append(grouped_id)

            for id in ids:
                if id not in regex_data.keys():
                    regex_data[id] = []
                regex_data[id].append("{} {}\n".format(name, data))

        for key, lines in regex_data.items():
            file_name = "{}.stm".format(key)
            regex_files[key] = file_name
            with open(file_name, "w") as f:
                for l in lines:
                    f.write(l)

        return regex_files

    def _make_ref(self, outpath):
        c = Corpus()
        c.load(tk.uncached_path(self.corpus_path))

        with open(outpath, "w") as f:
            for seg in c.segments():
                name = seg.fullname()
                orth = seg.orth

                words = orth.split()
                filtered_words = [self.map.get(w, w) for w in words]
                data = " ".join(filtered_words).lower()
                f.write("{} {}\n".format(name, data))

    def _convert_hyp_regex(self):
        with open(tk.uncached_path(self.ctm), "r") as f:
            transcriptions = collections.defaultdict(list)
            for line in f:
                if line.startswith(";;"):
                    full_name = line.split(" ")[1]  # second field contains full segment name
                    continue

                fields = line.split()
                if 5 <= len(fields) <= 6:
                    recording = fields[0]
                    start = float(fields[2])
                    word = fields[4]
                    word = self.map.get(word, word)
                    transcriptions[full_name].append((start, word))

            for recording, times_and_words in transcriptions.items():
                times_and_words.sort()

        regex_data = {}
        regex_files = {}
        for recording, times_and_words in transcriptions.items():
            data = " ".join([x[1] for x in times_and_words]).lower()
            res = re.match(self.regex, recording)

            ids = list(res.groups())
            grouped_id = "_".join(res.groups())
            ids.append(grouped_id)

            for id in ids:
                if id not in regex_data.keys():
                    regex_data[id] = []
                regex_data[id].append("{} {}\n".format(recording, data))

        for key, lines in regex_data.items():
            file_name = "{}.ctm".format(key)
            regex_files[key] = file_name
            with open(file_name, "w") as f:
                for l in lines:
                    f.write(l)

        return regex_files

    def _convert_hyp(self, outpath):
        with open(tk.uncached_path(self.ctm), "r") as f:
            transcriptions = collections.defaultdict(list)
            for line in f:
                if line.startswith(";;"):
                    full_name = line.split(" ")[1]  # second field contains full segment name
                    continue

                fields = line.split()
                if 5 <= len(fields) <= 6:
                    recording = fields[0]
                    start = float(fields[2])
                    word = fields[4]
                    word = self.map.get(word, word)
                    transcriptions[full_name].append((start, word))

            for recording, times_and_words in transcriptions.items():
                times_and_words.sort()

        with open(outpath, "w") as f:
            for recording, times_and_words in transcriptions.items():
                data = " ".join([x[1] for x in times_and_words]).lower()
                f.write("{} {}\n".format(recording, data))

    def run_regex(self):
        ref_reg = self._make_ref_regex()
        hyp_reg = self._convert_hyp_regex()

        exe = gs.KALDI_PATH + "/compute-wer"

        for key in ref_reg:
            ref_path = ref_reg[key]
            hyp_path = hyp_reg[key]

            report_path = "{}.report".format(key)

            with open(report_path, "w") as f:
                sp.run(
                    [
                        exe,
                        "--text",
                        "--mode=present",
                        "ark:" + ref_path,
                        "ark:" + hyp_path,
                    ],
                    stdout=f,
                )

        table_data = {}
        for f in os.listdir("."):
            if f.endswith(".report"):
                with open(f, "rt") as report_file:
                    for line in report_file:
                        if line.startswith("%WER"):
                            wer = float(line.split()[1])
                            table_data[f] = wer
                            break
            os.rename(f, os.path.join(self.out_report_dir.get_path(), f))

        with open(self.out_re_table.get_path(), "w") as f:
            for key, wer in table_data.items():
                f.write("{} {}\n".format(key, wer))

    def run(self, report_path=None, ref_path=None, hyp_path=None):
        if not report_path:
            report_path = self.out_report_path.get_path()
        if not ref_path:
            ref_path = self.out_kaldi_ref.get_path()
        if not hyp_path:
            hyp_path = self.out_kaldi_hyp.get_path()

        self._make_ref(ref_path)
        self._convert_hyp(hyp_path)

        exe = gs.KALDI_PATH + "/compute-wer"

        with open(report_path, "w") as f:
            sp.run(
                [exe, "--text", "--mode=present", "ark:" + ref_path, "ark:" + hyp_path],
                stdout=f,
            )

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(report_path="report.txt", ref_path="ref.txt", hyp_path="hyp.txt")
            with open("report.txt", "rt") as f:
                for line in f:
                    if line.startswith("%WER"):
                        wer = float(line.split()[1])
                        break
        os.chdir(old_dir)

        return wer


class IntersectStmCtm(Job):
    def __init__(self, stm_path: tk.Path, ctm_path: tk.Path):
        """
        This job calculate the intersection of a ctm and stm file and outputs new ctm and stm files, s.t. the recordings
        present in the ctm and stm file match. This can for example be used if a recognition corpus has been filtered
        and we want to compute the WER on the filtered corpus.

        :param stm_path: Path of the stm file
        :param ctm_path: Path of the ctm file
        """
        self.stm_path = stm_path
        self.ctm_path = ctm_path

        self.out_stm = self.output_path("intersected.stm")
        self.out_ctm = self.output_path("intersected.ctm")

        self.rqmt = None

    def tasks(self):
        yield Task("run", resume="run", mini_task=self.rqmt is None, rqmt=self.rqmt)

    def run(self):
        stm_lines = collections.defaultdict(list)
        ctm_lines = collections.defaultdict(list)

        with uopen(self.stm_path, "rt") as stm_in:
            for line in stm_in:
                if line.startswith(";"):
                    stm_lines[None].append(line)
                else:
                    recording = line.split()[0]
                    stm_lines[recording].append(line)

        segment_comments = []
        with uopen(self.ctm_path, "rt") as ctm_in:
            for line_num, line in enumerate(ctm_in, 1):
                if line_num == 1 and line.startswith(";; <name>"):  # header
                    ctm_lines[None].append(line)
                elif line.startswith(";"):
                    segment_comments.append(line)
                else:
                    recording = line.split()[0]
                    if segment_comments:
                        ctm_lines[recording].extend(segment_comments)
                        segment_comments = []
                    ctm_lines[recording].append(line)

        common_recordings = set(stm_lines.keys()).intersection(set(ctm_lines.keys()))
        del common_recordings[None]

        for path, lines in [(self.out_stm, stm_lines), (self.out_ctm, ctm_lines)]:
            with uopen(path, "wt") as out:
                out.write("".join(lines[None]))
                for r in common_recordings:
                    out.write("".join(lines[r]))
