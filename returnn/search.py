__all__ = [
    "ReturnnSearchJob",
    "ReturnnSearchJobV2",
    "ReturnnSearchFromFileJob",
    "SearchBPEtoWordsJob",
    "SearchOutputRawReplaceJob",
    "SearchWordsToCTMJob",
    "SearchWordsDummyTimesToCTMJob",
    "ReturnnComputeWERJob",
    "SearchRemoveLabelJob",
    "SearchCollapseRepeatedLabelsJob",
    "SearchBeamJoinScoresJob",
    "SearchTakeBestJob",
]

import copy
import logging
import os
import shutil
import subprocess as sp
from typing import Any, Optional, Union, Sequence, Set, Dict, Tuple

from sisyphus import *

from i6_core.deprecated.returnn_search import ReturnnSearchJob as _ReturnnSearchJob
from i6_core.lib.corpus import Corpus
import i6_core.util as util

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, PtCheckpoint

Path = setup_path(__package__)


class ReturnnSearchJob(_ReturnnSearchJob):
    """
    This Job was deprecated because absolute paths are hashed.

    See https://github.com/rwth-i6/i6_core/issues/274 for more details.
    """

    pass


class ReturnnSearchJobV2(Job):
    """
    Given a model checkpoint, run search task with RETURNN

    The outputs provided are:

     - out_search_file: Path to the Return search output in the provided format, which is "txt" (line based)
                        or "py" (python dictionary with seq_name to text mapping)

    """

    def __init__(
        self,
        search_data: Dict[str, Any],
        model_checkpoint: Union[Checkpoint, PtCheckpoint],
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        *,
        output_mode: str = "py",
        output_gzip: bool = False,
        log_verbosity: int = 3,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        """
        :param search_data: Returnn Dataset definition in dictionary form to be used for search
        :param model_checkpoint:  TF or PyTorch model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param tk.Path returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param tk.Path returnn_root: path to the RETURNN src folder
        :param str output_mode: "txt" or "py"
        :param bool output_gzip: if true, gzip the output file
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param time_rqmt: job time requirement in hours
        :param mem_rqmt: job memory requirement in GB
        :param cpu_rqmt: job cpu requirement in number of cores
        """
        assert isinstance(returnn_config, ReturnnConfig)
        kwargs = locals()
        del kwargs["self"]

        self.model_checkpoint = model_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_search_file = self.output_path("search_out" + (f".{output_mode}.gz" if output_gzip else ""))

        self.returnn_config = ReturnnSearchJobV2.create_returnn_config(**kwargs)
        self.returnn_config.post_config["search_output_file"] = self.out_search_file

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        config = self.returnn_config
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert self.model_checkpoint.exists(), "Provided model does not exists: %s" % str(self.model_checkpoint)

    def run(self):
        call = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
        sp.check_call(call, env=env)

    @classmethod
    def create_returnn_config(
        cls,
        search_data,
        model_checkpoint,
        returnn_config,
        output_mode,
        log_verbosity,
        device,
        **kwargs,
    ):
        """
        Creates search RETURNN config
        :param dict[str] search_data: dataset used for search
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param str output_mode: "txt" or "py"
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :rtype: ReturnnConfig
        """
        assert device in ["gpu", "cpu"]
        original_config = returnn_config.config
        res = copy.deepcopy(returnn_config)
        assert output_mode in ["py", "txt"]

        config = {
            "load": model_checkpoint,
            "search_output_file_format": output_mode,
            "need_data": False,
            "search_do_eval": 0,
        }

        config.update(res.config)  # update with the copy

        # override always
        config["task"] = "search"
        config["max_seq_length"] = 0

        if "search_data" in original_config:
            config["search_data"] = {
                **original_config["search_data"].copy(),
                **search_data,
            }
        else:
            config["search_data"] = search_data

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        post_config.update(res.post_config)

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnSearchJobV2.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        if kwargs.get("output_gzip", False) is not False:
            d["output_gzip"] = kwargs["output_gzip"]
        return super().hash(d)


class ReturnnSearchFromFileJob(Job):
    """
    Run search mode with a given RETURNN config.

    If the model path and epoch should be defined from the outside, please use
    `ext_model`for the model path and `ext_load_epoch` for the epoch in the config and parameter dict.

    As the existance of the model will be checked via `update`, it is possible to define checkpoints
    that do not exist yet, or checkpoint that can be automatically deleted during the training.

    """

    def __init__(
        self,
        returnn_config_file,
        parameter_dict=None,
        output_mode="py",
        default_model_name="epoch",
        time_rqmt=4,
        mem_rqmt=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param tk.Path returnn_config_file: a returnn training config file
        :param dict parameter_dict: provide external parameters to the rnn.py call
        :param str output_mode: "py" or "txt"
        :param float|int time_rqmt:
        :param float|int mem_rqmt:
        :param tk.Path|str returnn_python_exe: RETURNN python executable
        :param tk.Path|str returnn_root: RETURNN source root
        """

        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        self.returnn_config_file_in = returnn_config_file
        self.default_model_name = default_model_name
        self.parameter_dict = parameter_dict
        if self.parameter_dict is None:
            self.parameter_dict = {}

        self.returnn_config_file = self.output_path("returnn.config")

        self.rqmt = {"gpu": 1, "cpu": 2, "mem": mem_rqmt, "time": time_rqmt}

        assert output_mode in ["py", "txt"]
        self.out_search_file = self.output_path("search_results.%s" % output_mode)

        self.parameter_dict["search_output_file"] = self.out_search_file.get_path()
        self.parameter_dict["search_output_file_format"] = output_mode

    def update(self):
        if "ext_model" in self.parameter_dict and "ext_load_epoch" in self.parameter_dict:
            epoch = self.parameter_dict["ext_load_epoch"]
            epoch = epoch.get() if isinstance(epoch, tk.Variable) else epoch
            model_dir = self.parameter_dict["ext_model"]
            if isinstance(model_dir, tk.Path):
                self.add_input(
                    Path(
                        tk.uncached_path(model_dir) + "/%s.%03d.index" % (self.default_model_name, epoch),
                        creator=model_dir.creator,
                    )
                )
            else:
                self.add_input(Path(tk.uncached_path(model_dir) + "/%s.%03d.index" % (self.default_model_name, epoch)))

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def get_parameter_list(self):
        parameter_list = []
        for k, v in sorted(self.parameter_dict.items()):
            if isinstance(v, (tk.Variable, tk.Path)):
                v = v.get()
            elif isinstance(v, (list, dict, tuple)):
                v = '"%s"' % str(v).replace(" ", "")

            if isinstance(v, (float, int)) and v < 0:
                v = "+" + str(v)
            else:
                v = str(v)

            if k == "ext_model" and not v.endswith("/%s" % self.default_model_name):
                v = v + "/%s" % self.default_model_name

            parameter_list += ["++%s" % k, v]

        return parameter_list

    def create_files(self):
        # returnn
        shutil.copy(
            tk.uncached_path(self.returnn_config_file_in),
            tk.uncached_path(self.returnn_config_file),
        )

        parameter_list = self.get_parameter_list()

        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.returnn_config_file.get_path(),
        ] + parameter_list

        util.create_executable("rnn.sh", cmd)

    def run(self):
        sp.check_call(["./rnn.sh"])

    @classmethod
    def hash(cls, kwargs):
        kwargs = copy.copy(kwargs)
        kwargs.pop("default_model_name")
        kwargs.pop("time_rqmt")
        kwargs.pop("mem_rqmt")
        return super().hash(kwargs)


class SearchBPEtoWordsJob(Job):
    """
    converts BPE tokens to words in the python format dict from the returnn search
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: if True, gzip the output
        """
        self.search_py_output = search_py_output
        self.out_word_search_results = self.output_path("word_search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_word_search_results.get_path())
        with util.uopen(self.out_word_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in sorted(d.items()):
                if "#" in seq_tag:
                    tag_split = seq_tag.split("/")
                    recording_name, segment_name = tag_split[2].split("#")
                    seq_tag = tag_split[0] + "/" + recording_name + "/" + segment_name
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, text.replace("@@ ", "")))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, entry.replace("@@ ", "")))
            out.write("}\n")


class SearchOutputRawReplaceJob(Job):
    """
    Converts via replacement list.

    Generalizes over :class:`SearchBPEtoWordsJob`.
    BPE-to-words::

        words = SearchOutputRawReplaceJob(bpe, [("@@ ", "")], output_gzip=True).out_search_results

    SentencePiece-to-words::

        words = SearchOutputRawReplaceJob(spm, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results

    """

    def __init__(
        self, search_py_output: Path, replacement_list: Sequence[Tuple[str, str]], *, output_gzip: bool = False
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param replacement_list: list/sequence of (old, new) pairs to perform ``s.replace(old,new)`` on the raw text
        :param output_gzip: if True, gzip the output
        """
        self.search_py_output = search_py_output
        self.replacement_list = replacement_list
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_search_results.get_path())

        def _transform_text(s: str):
            for in_, out_ in self.replacement_list:
                s = s.replace(in_, out_)
            return s

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, _transform_text(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, _transform_text(entry)))
            out.write("}\n")


class SearchWordsToCTMJob(Job):
    """
    Convert RETURNN search output file into CTM format file (does not support n-best lists yet)
    """

    def __init__(self, recog_words_file, bliss_corpus, filter_tags=True):
        """
        :param Path recog_words_file: search output file from RETURNN
        :param Path bliss_corpus: bliss xml corpus
        :param bool filter_tags: if set to True, tags such as [noise] will be filtered out
        """
        self.recog_words_file = recog_words_file
        self.bliss_corpus = bliss_corpus
        self.filter_tags = filter_tags

        self.out_ctm_file = self.output_path("search.ctm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus = Corpus()
        corpus.load(self.bliss_corpus.get_path())
        d = util.parse_text_dict(self.recog_words_file)
        with util.uopen(self.out_ctm_file.get_path(), "wt") as out:
            # Do not print optional [n-best] header, some downstream evaluation pipelines
            # use the number of headers for validation. Since we do not print n-best-list
            # information this validation fails and discards the entire search outputs.
            #
            # See https://github.com/rwth-i6/i6_core/pull/542.
            out.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
            for seg in corpus.segments():
                seg_start = 0.0 if seg.start == float("inf") else seg.start
                seg_end = 0.0 if seg.end == float("inf") else seg.end
                seg_fullname = seg.fullname()
                assert seg_fullname in d, "can not find {} in search output".format(seg_fullname)
                out.write(";; %s (%f-%f)\n" % (seg_fullname, seg_start, seg_end))
                assert isinstance(d[seg_fullname], str), "no support for n-best lists yet"
                words = d[seg_fullname].split()
                # Just linearly interpolate the start/end of each word as time stamps are not given
                avg_dur = (seg_end - seg_start) * 0.9 / max(len(words), 1)
                count = 0
                for i in range(len(words)):
                    if self.filter_tags and words[i].startswith("[") and words[i].endswith("]"):
                        continue
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg.recording.name,
                            seg_start + avg_dur * i,
                            avg_dur,
                            words[i],
                        )
                    )
                    count += 1
                if count == 0:
                    # sclite cannot handle empty sequences, and would stop with an error like:
                    #   hyp file '4515-11057-0054' and ref file '4515-11057-0053' not synchronized
                    #   sclite: Alignment failed.  Exiting
                    # So we make sure it is never empty.
                    # For the WER, it should not matter, assuming the reference sequence is non-empty,
                    # you will anyway get a WER of 100% for this sequence.
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg.recording.name,
                            seg_start,
                            avg_dur,
                            "<empty-sequence>",
                        )
                    )


class SearchWordsDummyTimesToCTMJob(Job):
    """
    Convert RETURNN search output file into CTM format file (does not support n-best lists yet).
    Like :class:`SearchWordsToCTMJob` but does not use the Bliss XML corpus for recording names and segment times.
    Instead, this will just use dummy times.

    When creating the corresponding STM files, make sure it uses the same dummy times.
    """

    def __init__(
        self,
        recog_words_file: Path,
        *,
        seq_order_file: Optional[Path] = None,
        filter_tags: bool = True,
        seg_length_time: float = 1.0,
    ):
        """
        :param recog_words_file: search output file from RETURNN
        :param seq_order_file: file which defines the sequence order, i.e. the order of the segments in the CTM.
            This is required because sclite requires the same sequence order in the CTM as in the STM file,
            and the search output (recog_words_file) likely has a different order.
            Alternatively, you can use ``sort_files=True`` in the :class:`ScliteJob`,
            thus this is optional here.
            This file can be another text-dict format, e.g. via :class:`CorpusToTextDictJob`.
        :param filter_tags: if set to True, tags such as [noise] will be filtered out
        :param seg_length_time: dummy segment length time, in seconds
        """
        self.recog_words_file = recog_words_file
        self.seq_order_file = seq_order_file
        self.filter_tags = filter_tags
        self.seg_length_time = seg_length_time

        self.out_ctm_file = self.output_path("search.ctm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = util.parse_text_dict(self.recog_words_file)
        if self.seq_order_file is not None:
            seq_order = eval(util.uopen(self.seq_order_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(seq_order, (dict, list, tuple))
        else:
            seq_order = d.keys()
        with util.uopen(self.out_ctm_file.get_path(), "wt") as out:
            # Do not print optional [n-best] header, some downstream evaluation pipelines
            # use the number of headers for validation. Since we do not print n-best-list
            # information this validation fails and discards the entire search outputs.
            #
            # See https://github.com/rwth-i6/i6_core/pull/542.
            out.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
            for seg_fullname in seq_order:
                assert isinstance(
                    seg_fullname, str
                ), f"invalid seq_order entry {seg_fullname!r} (type {type(seg_fullname).__name__})"
                assert seg_fullname in d, f"seq_order entry {seg_fullname!r} not found in recog_words_file"
                text = d[seg_fullname]
                seg_start = 0.0
                seg_end = self.seg_length_time
                out.write(";; %s (%f-%f)\n" % (seg_fullname, seg_start, seg_end))
                words = text.split()
                time_step_per_word = (seg_end - seg_start) / max(len(words), 1)
                avg_dur = time_step_per_word * 0.9
                count = 0
                for i in range(len(words)):
                    if self.filter_tags and words[i].startswith("[") and words[i].endswith("]"):
                        continue
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg_fullname,  # originally the recording name, but treat each segment as a recording
                            seg_start + time_step_per_word * i,
                            avg_dur,
                            words[i],
                        )
                    )
                    count += 1
                if count == 0:
                    # sclite cannot handle empty sequences, and would stop with an error like:
                    #   hyp file '4515-11057-0054' and ref file '4515-11057-0053' not synchronized
                    #   sclite: Alignment failed.  Exiting
                    # So we make sure it is never empty.
                    # For the WER, it should not matter, assuming the reference sequence is non-empty,
                    # you will anyway get a WER of 100% for this sequence.
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg_fullname,
                            seg_start,
                            avg_dur,
                            "<empty-sequence>",
                        )
                    )


class ReturnnComputeWERJob(Job):
    """
    Computes WER using the calculate-word-error-rate.py tool from RETURNN
    """

    def __init__(self, hypothesis, reference, returnn_python_exe=None, returnn_root=None):
        """

        :param Path hypothesis: python-style search output from RETURNN (e.g. SearchBPEtoWordsJob)
        :param Path reference: python-style text dictionary (use e.g. i6_core.corpus.convert.CorpusToTextDictJob)
        :param str|Path returnn_python_exe: RETURNN python executable
        :param str|Path returnn_root: RETURNN source root
        """
        self.hypothesis = hypothesis
        self.reference = reference

        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        self.out_wer = self.output_var("wer")

    def run(self):
        call = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(str(self.returnn_root), "tools/calculate-word-error-rate.py"),
            "--expect_full",
            "--hyps",
            self.hypothesis.get_path(),
            "--refs",
            self.reference.get_path(),
            "--out",
            self.out_wer.get_path(),
        ]
        logging.info("run %s" % " ".join(call))
        sp.check_call(call)

    def tasks(self):
        yield Task("run", mini_task=True)


class SearchTakeBestJob(Job):
    """
    From RETURNN beam search results, extract the best result for each sequence.
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        :param output_gzip: if True, the output will be gzipped
        """
        self.search_py_output = search_py_output
        self.out_best_search_results = self.output_path("best_search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_best_search_results.get_path())
        with util.uopen(self.out_best_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                best_score, best_entry = max(entry)
                out.write("%r: %r,\n" % (seq_tag, best_entry))
            out.write("}\n")


class SearchRemoveLabelJob(Job):
    """
    Remove some labels from the search output, e.g. "<blank>".
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, remove_label: Union[str, Set[str]], output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param remove_label: label(s) to remove from the output, e.g. "<blank>"
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        if isinstance(remove_label, str):
            remove_label = {remove_label}
        assert isinstance(remove_label, set)
        self.remove_label = remove_label
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, self._filter(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, self._filter(entry)))
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t for t in tokens if t not in self.remove_label]
        return " ".join(tokens)


class SearchCollapseRepeatedLabelsJob(Job):
    """
    Collapse all repeated (white-space delimited) labels.
    """

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, self._filter(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, self._filter(entry)))
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t1 for (t1, t2) in zip(tokens, [None] + tokens) if t1 != t2]
        return " ".join(tokens)


class SearchBeamJoinScoresJob(Job):
    """
    Expects a beam of hypotheses.
    If there are multiple hyps which are the same (e.g. this can happen after "<blank>" removal),
    it will collapse them into a single hyp with the logsumexp of the scores.
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: if True, will gzip the output
        """
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import numpy

        neg_inf = -float("inf")

        def logsumexp(*args):
            """
            Stable log sum exp.
            """
            if all(a == neg_inf for a in args):
                return neg_inf
            a_max = max(args)
            lsp = numpy.log(sum(numpy.exp(a - a_max) for a in args))
            return a_max + lsp

        d = util.parse_text_dict(self.search_py_output)
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                # n-best list as [(score, text), ...]
                assert isinstance(entry, list)
                hyps = {}  # text -> score
                for score, text in entry:
                    if text not in hyps:
                        hyps[text] = score
                    else:
                        hyps[text] = logsumexp(hyps[text], score)
                out.write("%r: [\n" % (seq_tag,))
                for score, text in sorted([(score, text) for text, score in hyps.items()], reverse=True):
                    out.write("(%f, %r),\n" % (score, text))
                out.write("],\n")
            out.write("}\n")
