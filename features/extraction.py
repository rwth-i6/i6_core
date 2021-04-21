__all__ = ["FeatureExtractionJob"]

import copy
import shutil

from sisyphus import *

Path = setup_path(__package__)

from .common import *
import recipe.i6_core.rasr as rasr
import recipe.i6_core.util as util


class FeatureExtractionJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        port_name_mapping,
        one_dimensional_outputs=None,
        job_name="features",
        rtf=0.1,
        mem=2,
        extra_config=None,
        extra_post_config=None,
    ):
        """
        :param recipe.rasr.crp.CommonRasrParameters crp:
        :param recipe.rasr.flow.FlowNetwork feature_flow:
        :param dict[str,str] port_name_mapping:
        :param set[str]|None one_dimensional_outputs:
        :param str job_name:
        :param float rtf:
        :param int mem:
        :param recipe.rasr.config.RasrConfig|None extra_config:
        :param recipe.rasr.config.RasrConfig|None extra_post_config:
        """
        self.set_vis_name("Extract %s" % job_name)

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = FeatureExtractionJob.create_config(**kwargs)
        self.feature_flow = feature_flow
        self.cached_feature_flow = feature_extraction_cache_flow(
            feature_flow, port_name_mapping, one_dimensional_outputs
        )
        self.exe = (
            crp.feature_extraction_exe
            if crp.feature_extraction_exe is not None
            else self.default_exe("feature-extraction")
        )
        self.concurrent = crp.concurrent

        self.log_file = self.log_file_output_path("feature-extraction", crp, True)
        self.single_feature_caches = {}
        self.feature_bundle = {}
        self.feature_path = {}
        for name in set(port_name_mapping.values()):
            self.single_feature_caches[name] = dict(
                (
                    task_id,
                    self.output_path("%s.cache.%d" % (name, task_id), cached=True),
                )
                for task_id in range(1, crp.concurrent + 1)
            )
            self.feature_bundle[name] = self.output_path(
                "%s.cache.bundle" % name, cached=True
            )
            self.feature_path[name] = util.MultiOutputPath(
                self,
                "%s.cache.$(TASK)" % name,
                self.single_feature_caches[name],
                cached=True,
            )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 1,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])
        for name in self.single_feature_caches:
            shutil.move(
                "%s.cache.%d" % (name, task_id),
                self.single_feature_caches[name][task_id].get_path(),
            )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("feature-extraction.log.%d" % task_id)
        for name in self.feature_bundle:
            util.delete_if_zero("%s.cache.%d" % (name, task_id))

    def create_files(self):
        self.write_config(self.config, self.post_config, "feature-extraction.config")
        self.cached_feature_flow.write_to_file("feature-extraction.flow")
        for name in self.feature_bundle:
            util.write_paths_to_file(
                self.feature_bundle[name], self.single_feature_caches[name].values()
            )
        self.write_run_script(self.exe, "feature-extraction.config")

    @classmethod
    def create_config(
        cls, crp, feature_flow, extra_config, extra_post_config, **kwargs
    ):
        """
        :param recipe.rasr.crp.CommonRasrParameters crp:
        :param feature_flow:
        :param recipe.rasr.config.RasrConfig|None extra_config:
        :param recipe.rasr.config.RasrConfig|None extra_post_config:
        :return: config, post_config
        :rtype: (recipe.rasr.config.RasrConfig, recipe.rasr.config.RasrConfig)
        """
        config, post_config = rasr.build_config_from_mapping(
            crp, {"corpus": "extraction.corpus"}, parallelize=True
        )
        config.extraction.feature_extraction.file = "feature-extraction.flow"
        # this was a typo but we cannot remove it now without breaking a lot of hashes
        config.extraction.feature_etxraction["*"].allow_overwrite = True
        # thus we 'fix' it in post_config
        post_config.extraction.feature_extraction["*"].allow_overwrite = True
        post_config.extraction.feature_etxraction["*"].allow_overwrite = None

        feature_flow.apply_config("extraction.feature-extraction", config, post_config)

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return (
            kwargs["job_name"]
            + "."
            + super().hash(
                {
                    "config": config,
                    "flow": kwargs["feature_flow"],
                    "exe": kwargs["crp"].feature_extraction_exe,
                }
            )
        )
