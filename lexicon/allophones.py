__all__ = ["StoreAllophones", "DumpStateTying"]

import shutil

from sisyphus import *

Path = setup_path(__package__)

import recipe.i6_asr.rasr as rasr
import recipe.i6_asr.util as util


class StoreAllophones(rasr.RasrCommand, Job):
    def __init__(
        self,
        csp,
        num_single_state_monophones=1,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Store Allophones")

        self.config, self.post_config = StoreAllophones.create_config(
            csp, extra_config, extra_post_config
        )
        self.exe = self.select_exe(csp.allophone_tool_exe, "allophone-tool")
        self.num_single_state_monophones = (
            num_single_state_monophones  # usually only silence and noise
        )

        self.log_file = self.log_file_output_path("store-allophones", csp, False)
        self.allophone_file = self.output_path("allophones")
        self.num_allophones = self.output_var("num_allophones")
        self.num_monophones = self.output_var("num_monophones")
        self.num_monophone_states = self.output_var("num_monophone_states")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "store-allophones.config")
        self.write_run_script(self.exe, "store-allophones.config")

    def run(self):
        self.run_script(1, self.log_file)
        shutil.move("allophones", self.allophone_file.get_path())

        with open(self.allophone_file.get_path(), "rt") as f:
            allophones = f.readlines()[1:]

        self.num_allophones.set(len(allophones))

        num_monophones = len(set(a.split("{")[0] for a in allophones))
        self.num_monophones.set(num_monophones)

        self.config._update(
            self.post_config
        )  # make it easier to access states-per-phone
        states_per_phone = (
            self.config.allophone_tool.acoustic_model.hmm.states_per_phone
        )
        num_monophone_states = (
            self.num_single_state_monophones
            + (num_monophones - self.num_single_state_monophones) * states_per_phone
        )
        self.num_monophone_states.set(num_monophone_states)

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("store-allophones.log")

    @classmethod
    def create_config(cls, csp, extra_config, extra_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            csp,
            {
                "acoustic_model": "allophone-tool.acoustic-model",
                "lexicon": "allophone-tool.lexicon",
            },
            parallelize=False,
        )

        config.allophone_tool.acoustic_model.allophones.store_to_file = "allophones"

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["csp"].allophone_tool_exe})


class DumpStateTying(rasr.RasrCommand, Job):
    def __init__(self, csp, extra_config=None, extra_post_config=None):
        self.set_vis_name("Dump state-tying")

        self.config, self.post_config = DumpStateTying.create_config(
            csp, extra_config, extra_post_config
        )
        self.exe = self.select_exe(csp.allophone_tool_exe, "allophone-tool")

        self.log_file = self.log_file_output_path("dump-state-tying", csp, False)
        self.state_tying = self.output_path("state-tying")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "dump-state-tying.config")
        self.write_run_script(self.exe, "dump-state-tying.config")

    def run(self):
        self.run_script(1, self.log_file)
        shutil.move("state-tying", self.state_tying.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("dump-state-tying.log")

    @classmethod
    def create_config(cls, csp, extra_config, extra_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            csp,
            {
                "acoustic_model": "allophone-tool.acoustic-model",
                "lexicon": "allophone-tool.lexicon",
            },
            parallelize=False,
        )

        config.allophone_tool.dump_state_tying.channel = "state-tying-channel"
        config.allophone_tool.channels.state_tying_channel.append = False
        config.allophone_tool.channels.state_tying_channel.compressed = False
        config.allophone_tool.channels.state_tying_channel.file = "state-tying"
        config.allophone_tool.channels.state_tying_channel.unbuffered = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["csp"].allophone_tool_exe})
