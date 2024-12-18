__all__ = ["ComputePerplexityJob", "ComputePerplexityJobV2"]

from sisyphus import *

Path = setup_path(__package__)

import gzip
import shutil
import xml.etree.ElementTree as ET

import i6_core.rasr as rasr
import i6_core.util as util


class ComputePerplexityJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        text_file,
        encoding="utf-8",
        renormalize=False,
        extra_config=None,
        extra_post_config=None,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.text_file = text_file
        self.renormalize = renormalize

        self.config, self.post_config = ComputePerplexityJob.create_config(**kwargs)
        self.exe = self.select_exe(crp.lm_util_exe, "lm-util")

        self.log_file = self.log_file_output_path("compute_ppl", crp, False)
        self.score_file = self.output_path("word.scores")
        self.num_tokens = self.output_var("num-tokens")
        self.num_unks = self.output_var("num-unks")
        self.unk_ratio = self.output_var("unk-ratio")
        self.perplexity = self.output_var("perplexity")
        self.perplexity_without_eos = self.output_var("perplexity-without-eos")
        self.perplexity_without_unknowns = self.output_var("perplexity-without-unknowns")
        self.perplexity_without_eos_without_unknowns = self.output_var("perplexity-without-eos-without-unknowns")

        self.rqmt = {"time": 1, "cpu": 1, "mem": 2}

    def tasks(self):
        yield Task("create_files", resume="create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "compute_ppl.config")
        device = "gpu" if self.rqmt.get("gpu", 0) > 0 else "cpu"
        extra_code = (
            f":${{THEANO_FLAGS:="
            '}\nexport THEANO_FLAGS="$THEANO_FLAGS,device={device},force_device=True"\nexport TF_DEVICE="{device}"'
        )
        self.write_run_script(self.exe, "compute_ppl.config", extra_code=extra_code)

    def run(self):
        self.run_script(1, self.log_file)
        shutil.move("word.scores", self.score_file.get_path())

        open_fun = gzip.open if self.log_file.get_path().endswith(".gz") else open
        with open_fun(self.log_file.get_path(), "rt") as f:
            root = ET.parse(f)
        ppl = root.find(".//perplexity")
        self.perplexity.set(float(ppl.text))

        # Retrocompatibility checks
        ppl_wo_eos = root.find(".//perplexity-without-eos")
        if ppl_wo_eos is not None:
            self.perplexity_without_eos.set(float(ppl_wo_eos.text))
        else:
            self.perplexity_without_eos.set(float("inf"))

        ppl_wo_unks = root.find(".//perplexity-without-unknowns")
        if ppl_wo_unks is not None:
            self.perplexity_without_unknowns.set(float(ppl_wo_unks.text))
        else:
            self.perplexity_without_unknowns.set(float("inf"))

        ppl_wo_eos_wo_unks = root.find(".//perplexity-without-eos-without-unknowns")
        if ppl_wo_eos_wo_unks is not None:
            self.perplexity_without_eos_without_unknowns.set(float(ppl_wo_eos_wo_unks.text))
        else:
            self.perplexity_without_eos_without_unknowns.set(float("inf"))

        num_tokens = root.find(".//num-tokens")
        if num_tokens is not None:
            self.num_tokens.set(float(num_tokens.text))
        else:
            self.num_tokens.set(float("inf"))

        num_unks = root.find(".//num-unks")
        if num_unks is not None:
            self.num_unks.set(float(num_unks.text))
        else:
            self.num_unks.set(float("inf"))

        ratio_unks = root.find(".//unk-ratio")
        if ratio_unks is not None:
            self.unk_ratio.set(float(ratio_unks.text))
        else:
            self.unk_ratio.set(float("inf"))

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("compute_ppl.log")

    @classmethod
    def create_config(
        cls,
        crp,
        text_file,
        encoding,
        renormalize,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp, {"lexicon": "lm-util.lexicon", "language_model": "lm-util.lm"}
        )
        del config.lm_util.lm.scale  # scale not considered here, delete to remove ambiguity

        config.lm_util.action = "compute-perplexity-from-text-file"
        config.lm_util.file = text_file
        config.lm_util.encoding = encoding
        config.lm_util.batch_size = 100
        config.lm_util.renormalize = renormalize
        config.lm_util.score_file = "word.scores"

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config


class ComputePerplexityJobV2(ComputePerplexityJob):
    """
    Update of ComputePerplexityJob with a custom hash function that only hashes relevant inputs.
    Previous version will change hash even if unrelated components in the CommonRasrParameters object
    change their value, e.g. the acoustic-model.
    """

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].lm_util_exe})
