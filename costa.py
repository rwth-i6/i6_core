from sisyphus import *

Path = setup_path(__package__)

from recipe.i6_asr.features.common import raw_audio_flow
import recipe.i6_asr.rasr as rasr
import recipe.i6_asr.util as util


class CostaJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        csp,
        eval_recordings=False,
        eval_lm=True,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Costa")

        self.config, self.post_config = CostaJob.create_config(
            csp, eval_recordings, eval_lm, extra_config, extra_post_config
        )
        self.audio_flow = raw_audio_flow(csp.audio_format) if eval_recordings else None

        self.log_file = self.output_path(
            "costa.log" + (".gz" if csp.compress_log_file else "")
        )
        self.exe = (
            csp.costa_exe if csp.costa_exe is not None else self.default_exe("costa")
        )
        self.rqmt = {
            "time": max(csp.corpus_duration / 20, 0.5),
            "cpu": 1,
            "mem": 1 if not self.config.costa.lm_statistics else 4,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "costa.config")

        if self.audio_flow is not None:
            self.audio_flow.write_to_file("audio.flow")

        self.write_run_script(self.exe, "costa.config")

    def run(self):
        self.run_script(1, self.log_file)

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("costa.log")

    @classmethod
    def create_config(
        cls, csp, eval_recordings, eval_lm, extra_config, extra_post_config
    ):
        config = rasr.RasrConfig()
        post_config = rasr.RasrConfig()

        config._update(csp.log_config)
        post_config._update(csp.log_post_config)

        config.costa.statistics.corpus = csp.corpus_config
        post_config.costa.statistics.corpus = csp.corpus_post_config

        config.costa.statistics.evaluate_recordings = eval_recordings
        if eval_recordings:
            config.costa.statistics.feature_extraction.file = "audio.flow"

        config.costa.lexical_statistics = csp.lexicon_config is not None
        config.costa.statistics.lexicon = csp.lexicon_config
        post_config.costa.statistics.lexicon = csp.lexicon_post_config

        config.costa.lm_statistics = (
            csp.language_model_config is not None
            and csp.lexicon_config is not None
            and eval_lm
        )
        config.costa.statistics.lm = csp.language_model_config
        post_config.costa.statistics.lm = csp.language_model_post_config

        if extra_config is not None:
            config._update(extra_config)

        if extra_post_config is not None:
            config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["csp"].costa_exe})
