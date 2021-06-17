__all__ = [
    "AnalogJob",
    "ScliteJob",
    "Hub5ScoreJob",
    "QuaeroScorerJob",
    "KaldiScorerJob",
]

import os
import shutil
import subprocess as sp
import tempfile
import collections
import re

from sisyphus import *
from i6_core.lib.corpus import *

Path = setup_path(__package__)


class AnalogJob(Job):
    def __init__(self, configs, merge=True):
        self.set_vis_name("Analog")

        self.merge = merge
        self.configs = configs
        if type(configs) == dict:
            self.configs = list(configs.values())
        self.report = self.output_path("report.analog")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        analog_path = os.path.join(gs.RASR_ROOT, "src/Tools/Analog/analog")
        with open(self.report.get_path(), "w") as out:
            sp.check_call(
                [analog_path]
                + (["-m"] if self.merge else [])
                + [tk.uncached_path(c) for c in self.configs],
                stdout=out,
            )


class ScliteJob(Job):
    def __init__(self, ref, hyp, cer=False, sort_files=False, additional_args=None):
        self.set_vis_name("Sclite - %s" % ("CER" if cer else "WER"))

        self.ref = ref
        self.hyp = hyp
        self.cer = cer
        self.sort_files = sort_files
        self.additional_args = additional_args

        self.report_dir = self.output_path("reports", True)

        self.wer = self.output_var("wer")
        self.num_errors = self.output_var("num_errors")
        self.percent_correct = self.output_var("percent_correct")
        self.num_correct = self.output_var("num_correct")
        self.percent_substitution = self.output_var("percent_substitution")
        self.num_substitution = self.output_var("num_substitution")
        self.percent_deletions = self.output_var("percent_deletions")
        self.num_deletions = self.output_var("num_deletions")
        self.percent_insertions = self.output_var("percent_insertions")
        self.num_insertions = self.output_var("num_insertions")
        self.percent_word_accuracy = self.output_var("percent_word_accuracy")
        self.ref_words = self.output_var("ref_words")
        self.hyp_words = self.output_var("hyp_words")
        self.aligned_words = self.output_var("aligned_words")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self, output_to_report_dir=True):
        if self.sort_files:
            sort_stm_args = ["sort", "-k1,1", "-k4,4n", tk.uncached_path(self.ref)]
            (fd_stm, tmp_stm_file) = tempfile.mkstemp(suffix=".stm")
            res = sp.run(sort_stm_args, stdout=sp.PIPE)
            os.write(fd_stm, res.stdout)
            os.close(fd_stm)

            sort_ctm_args = ["sort", "-k1,1", "-k3,3n", tk.uncached_path(self.hyp)]
            (fd_ctm, tmp_ctm_file) = tempfile.mkstemp(suffix=".ctm")
            res = sp.run(sort_ctm_args, stdout=sp.PIPE)
            os.write(fd_ctm, res.stdout)
            os.close(fd_ctm)

        sclite_path = (
            os.path.join(gs.SCTK_PATH, "sclite")
            if hasattr(gs, "SCTK_PATH")
            else "sclite"
        )
        output_dir = self.report_dir.get_path() if output_to_report_dir else "."
        stm_file = tmp_stm_file if self.sort_files else tk.uncached_path(self.ref)
        ctm_file = tmp_ctm_file if self.sort_files else tk.uncached_path(self.hyp)

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
            with open(f"{output_dir}/sclite.dtl", "rt") as f:
                for line in f:
                    s = line.split()
                    if line.startswith("Percent Total Error"):
                        self.wer.set(float(s[4][:-1]))
                        self.num_errors.set(int("".join(s[5:])[1:-1]))
                    elif line.startswith("Percent Correct"):
                        self.percent_correct.set(float(s[3][:-1]))
                        self.num_correct.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Substitution"):
                        self.percent_substitution.set(float(s[3][:-1]))
                        self.num_substitution.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Deletions"):
                        self.percent_deletions.set(float(s[3][:-1]))
                        self.num_deletions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Insertions"):
                        self.percent_insertions.set(float(s[3][:-1]))
                        self.num_insertions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Word Accuracy"):
                        self.percent_word_accuracy.set(float(s[4][:-1]))
                    elif line.startswith("Ref. words"):
                        self.ref_words.set(int("".join(s[3:])[1:-1]))
                    elif line.startswith("Hyp. words"):
                        self.hyp_words.set(int("".join(s[3:])[1:-1]))
                    elif line.startswith("Aligned words"):
                        self.aligned_words.set(int("".join(s[3:])[1:-1]))

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(output_to_report_dir=False)
            dtl_file = "sclite.dtl"
            with open(dtl_file, "rt") as f:
                for line in f:
                    if line.startswith("Percent Total Error"):
                        errors = float("".join(line.split()[5:])[1:-1])
                    if line.startswith("Ref. words"):
                        wer = 100.0 * errors / float("".join(line.split()[3:])[1:-1])
                        break
        os.chdir(old_dir)

        return wer


class Hub5ScoreJob(Job):
    def __init__(self, ref, glm, hyp):
        self.set_vis_name("HubScore")

        self.glm = glm
        self.hyp = hyp
        self.ref = ref

        self.report_dir = self.output_path("reports", True)

        self.wer = self.output_var("wer")
        self.num_errors = self.output_var("num_errors")
        self.percent_correct = self.output_var("percent_correct")
        self.num_correct = self.output_var("num_correct")
        self.percent_substitution = self.output_var("percent_substitution")
        self.num_substitution = self.output_var("num_substitution")
        self.percent_deletions = self.output_var("percent_deletions")
        self.num_deletions = self.output_var("num_deletions")
        self.percent_insertions = self.output_var("percent_insertions")
        self.num_insertions = self.output_var("num_insertions")
        self.percent_word_accuracy = self.output_var("percent_word_accuracy")
        self.ref_words = self.output_var("ref_words")
        self.hyp_words = self.output_var("hyp_words")
        self.aligned_words = self.output_var("aligned_words")

        self.swb_num_errors = self.output_var("swb_num_errors")
        self.swb_ref_words = self.output_var("swb_ref_words")
        self.ch_num_errors = self.output_var("ch_num_errors")
        self.ch_ref_words = self.output_var("ch_ref_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self, move_files=True):
        hubscr_path = (
            os.path.join(gs.SCTK_PATH, "hubscr.pl")
            if hasattr(gs, "SCTK_PATH")
            else "hubscr.pl"
        )
        sctk_opt = ["-p", gs.SCTK_PATH] if hasattr(gs, "SCTK_PATH") else []

        ref = self.ref
        try:
            ref = shutil.copy(tk.uncached_path(ref), ".")
        except shutil.SameFileError:
            pass

        hyp = self.hyp
        try:
            hyp = shutil.copy(tk.uncached_path(hyp), ".")
        except shutil.SameFileError:
            pass

        sp.check_call(
            [hubscr_path, "-V", "-l", "english", "-h", "hub5"]
            + sctk_opt
            + ["-g", tk.uncached_path(self.glm), "-r", ref, hyp]
        )

        if move_files:  # run as real job
            with open(f"{hyp}.filt.dtl", "rt") as f:
                for line in f:
                    s = line.split()
                    if line.startswith("Percent Total Error"):
                        self.wer.set(float(s[4][:-1]))
                        self.num_errors.set(int("".join(s[5:])[1:-1]))
                    elif line.startswith("Percent Correct"):
                        self.percent_correct.set(float(s[3][:-1]))
                        self.num_correct.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Substitution"):
                        self.percent_substitution.set(float(s[3][:-1]))
                        self.num_substitution.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Deletions"):
                        self.percent_deletions.set(float(s[3][:-1]))
                        self.num_deletions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Insertions"):
                        self.percent_insertions.set(float(s[3][:-1]))
                        self.num_insertions.set(int("".join(s[4:])[1:-1]))
                    elif line.startswith("Percent Word Accuracy"):
                        self.percent_word_accuracy.set(float(s[4][:-1]))
                    elif line.startswith("Ref. words"):
                        self.ref_words.set(int(s[3][1:-1]))
                    elif line.startswith("Hyp. words"):
                        self.hyp_words.set(int(s[3][1:-1]))
                    elif line.startswith("Aligned words"):
                        self.aligned_words.set(int(s[3][1:-1]))

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

            self.swb_num_errors.set(swb_err)
            self.swb_ref_words.set(swb_ref)
            self.ch_num_errors.set(ch_err)
            self.ch_ref_words.set(ch_ref)

            for f in os.listdir("."):
                os.rename(f, os.path.join(self.report_dir.get_path(), f))

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

        self.report_dir = self.output_path("reports", True)

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
                os.rename(f, os.path.join(self.report_dir.get_path(), f))

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

        self.kaldi_ref = self.output_path("ref.txt")
        self.kaldi_hyp = self.output_path("hyp.txt")
        self.report_dir = self.output_path("reports", True)
        self.report_path = self.output_path("reports/wer.txt")
        if regex:
            self.re_table = self.output_path("reports/table.txt")

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
                    full_name = line.split(" ")[
                        1
                    ]  # second field contains full segment name
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
                    full_name = line.split(" ")[
                        1
                    ]  # second field contains full segment name
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
            os.rename(f, os.path.join(self.report_dir.get_path(), f))

        with open(self.re_table.get_path(), "w") as f:
            for key, wer in table_data.items():
                f.write("{} {}\n".format(key, wer))

    def run(self, report_path=None, ref_path=None, hyp_path=None):
        if not report_path:
            report_path = self.report_path.get_path()
        if not ref_path:
            ref_path = self.kaldi_ref.get_path()
        if not hyp_path:
            hyp_path = self.kaldi_hyp.get_path()

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
