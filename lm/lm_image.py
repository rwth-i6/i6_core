__all__ = ["LmImageJob"]

import shutil

from sisyphus import *
from i6_core import rasr
from i6_core import util

Path = setup_path(__package__)


class LmImageJob(rasr.RasrCommand, Job):
    """
    pre-compute LM image without generating global cache
    """

    def __init__(
        self,
        crp,
        extra_config=None,
        extra_post_config=None,
        encoding="utf-8",
        renormalize=False,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.text_file = text_file
        self.renormalize = renormalize

        self.config, self.post_config, self.num_images = LmImageJob.create_config(
            **kwargs
        )
        self.exe = self.select_exe(crp.lm_util_exe, "lm-util")

        self.log_file = self.log_file_output_path("lm_image", crp, False)
        self.out_lm_images = {
            i: self.output_path(f"lm-{i}.image", cached=True)
            for i in range(1, self.num_images + 1)
        }

        self.rqmt = {"time": 1, "cpu": 1, "mem": 2}

    def tasks(self):
        yield Task("create_files", resume="create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "lm_image.config")
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format(
                "gpu" if self.rqmt.get("gpu", 0) > 0 else "cpu"
            )
        )
        self.write_run_script(self.exe, "lm_image.config", extra_code=extra_code)

    def run(self):
        self.run_script(1, self.log_file)
        for i in range(1, self.num_images + 1):
            shutil.move(f"lm-{i}.image", self.lm_images[i].get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("lm_image.log")

    @classmethod
    def find_arpa_lms(cls, lm_config, lm_post_config=None):
        result = []

        def has_image(c, pc):
            res = c._get("image") is not None
            res = res or (pc is not None and pc._get("image") is not None)
            return res

        if lm_config.type == "ARPA":
            if not has_image(lm_config, lm_post_config):
                result.append((lm_config, lm_post_config))
        elif lm_config.type == "combine":
            for i in range(1, lm_config.num_lms + 1):
                sub_lm_config = lm_config["lm-%d" % i]
                sub_lm_post_config = (
                    lm_post_config["lm-%d" % i] if lm_post_config is not None else None
                )
                result += cls.find_arpa_lms(sub_lm_config, sub_lm_post_config)
        return result

    @classmethod
    def create_config(
        cls,
        crp,
        extra_config,
        extra_post_config,
        encoding,
        renormalize,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp, {"lexicon": "lm-util.lexicon", "language_model": "lm-util.lm"}
        )
        del (
            config.lm_util.lm.scale
        )  # scale not considered here, delete to remove ambiguity

        config.lm_util.action = "load-lm"
        config.lm_util.file = crp.language_model_config.file
        config.lm_util.encoding = encoding
        config.lm_util.batch_size = 100
        config.lm_util.renormalize = renormalize

        config._update(extra_config)
        post_config._update(extra_post_config)

        arpa_lms = cls.find_arpa_lms(
            config,
            post_config if post_config else None,
        )

        return config, post_config, len(arpa_lms)
