import copy
from collections import defaultdict
from enum import Enum
import os
import pathlib
import subprocess as sp
from typing import Dict, Any, Optional
import yaml

from sisyphus import Task, Job, tk
from sisyphus.hash import sis_hash_helper
import sisyphus.global_settings as gs

import i6_core.util as util


class FairseqHydraConfig:
    """
    An object that manages a Fairseq hydra config.
    """

    def __init__(
        self,
        config_dict: Dict[str, Any],
        post_config_dict: Optional[Dict[str, Any]] = None,
        *,
        package_name: str = "",
    ):
        """
        :param config_dict: Contains the information which is needed for fairseq-hydra-train. Will be converted and dumped into a .yaml
        :param dict post_config_dict: dictionary of the FairseqHydraConfig config variables that are not hashed
        :param package_name: The @package directory that is required to be added to the top of Hydra config, for example "# @package _group_"
        """
        assert isinstance(config_dict, dict)
        assert isinstance(post_config_dict, dict) or post_config_dict is None
        self.config_dict = config_dict
        self.post_config_dict = post_config_dict if post_config_dict is not None else {}
        self.package_name = package_name
        self.check_consistency()

    @property
    def data(self):
        """
        Get the underlying data of the FairseqHydraConfig
        """
        return self.config_dict

    def write(self, path: str):
        config_dict = self.config_dict.copy()
        config_dict = util.update_nested_dict(config_dict, self.post_config_dict)

        # recursively go through config dictionary to get all sisyphus paths inplace
        config_dict = util.instanciate_delayed(config_dict)

        config_yaml = yaml.dump(config_dict)
        if self.package_name != "":
            config_yaml = f"# @package {self.package_name}\n" + config_yaml
        with open(path, "w") as file:
            file.write(config_yaml)

    def update(self, other):
        """
        updates a FairseqHydraConfig with another FairseqHydraConfig:
          * config_dict, post_config_dict use dict.update
        :param FairseqHydraConfig other:
        """
        self.config_dict = util.update_nested_dict(self.config_dict, other.config_dict)
        self.post_config_dict = util.update_nested_dict(self.post_config_dict, other.post_config_dict)
        if other.package_name != "":
            self.package_name = other.package_name
        self.check_consistency()

    def check_consistency(self):
        """
        Check that there is no config key overwritten by post_config.
        Also check for parameters that should never be hashed.
        """
        for group in self.config_dict:
            if isinstance(self.config_dict[group], dict):
                for key in self.config_dict[group]:
                    if group in self.post_config_dict:
                        assert (
                            key not in self.post_config_dict[group].keys()
                        ), f"{key} of {group} in post_config would overwrite existing entry in config"
            else:
                assert (
                    group not in self.post_config_dict
                ), f"{group} in post_config would overwrite existing entry in config"

        # list of parameters that should never be hashed
        disallowed_in_config = ["save_interval", "max_epoch", "max_update"]

        for group in self.config_dict:
            if isinstance(self.config_dict[group], dict):
                for key in disallowed_in_config:
                    assert (
                        self.config_dict[group].get(key) is None
                    ), f"please define {key} of {group} only as parameter in the post_config_dict"
            else:
                assert (
                    self.config_dict[group].get(group) is None
                ), f"please define {group} only as parameter in the post_config_dict"

    def _sis_hash(self):
        h = {"fairseq_hydra_config": self.config_dict}
        return sis_hash_helper(h)


class PytorchHydraModel:
    """
    Defines a Pytorch hydra model as yaml config, pytorch checkpoint file and epoch
    """

    def __init__(self, fairseq_hydra_config_file: tk.Path, model: tk.Path, epoch: int):
        """
        :param fairseq_hydra_config_file: Path to a returnn config file
        :param model: Path to a pytorch checkpoint
        :param epoch: Number of epochs this model was trained
        """
        self.returnn_config_file = fairseq_hydra_config_file
        self.model = model
        self.epoch = epoch


class CacheManagerType(Enum):
    none = 1
    i6 = 2
    general = 3


class FairseqHydraTrainingJob(Job):
    """
    Train a Fairseq model using fairseq-hydra-train
    """

    def __init__(
        self,
        fairseq_hydra_config,
        *,  # args below are keyword only
        command_line_args=None,
        max_epoch=1,
        max_update=1,
        save_interval=1,
        keep_epochs=None,
        rqmt=None,
        fairseq_python_exe=None,
        fairseq_root,
        cache_manager=CacheManagerType.none,
        zipped_audio_dir=None,
    ):
        """
        :param FairseqHydraConfig fairseq_hydra_config: Fairseq hydra config
        :param list command_line_args: Additional command line arguments (starting with "--*"),
            to configure the Fairseq-hydra task
        :param int max_epoch: maximum number of epochs to run.
        :param int max_update: maximum number of steps to run.
        :param int save_interval: save a checkpoint each n-th epoch
        :param list[int]|set[int]|None keep_epochs: specify which checkpoints are kept in self.out_models.
            Use None for each save_interval-th epoch
        :param dict[str, int|float rqmt: the resource requirement including
            the overall time requirements, i.e. intime_rqmt,
            the memory requirements (per GPU), i.e. mem_rqmt
            the required number of CPUs (per GPU), i.e. cpu_rqmt
            the number of required GPUs gpu_rqmt, i.e. gpu_rqmt
        :param tk.Path fairseq_python_exe: File path to the executable for running python
        :param tk.Path fairseq_root: File path to the fairseq git for alternative call of fairseq-hydra-train
            (no need to install fairseq here)
        :param enum cache_manager: if not CacheManagerType.none, enables caching of data given in the manifest with cache manager
            possible values: CacheManagerType.none: no caching, CacheManagerType.i6: use the i6 specific cache manager,
            CacheManagerType.general: apply gs.file_caching
        :param [tk.Path]|tk.Path zipped_audio_dir: using a bundle file for caching is very slow for large manifests. For
            speeding up the audio file transfer using the cache manager, a zipped audio directory might be provided.
            The zipped audio directory is then used for caching instead and unzipped on the node for training
        """

        # Inputs:
        kwargs = locals()
        del kwargs["self"]

        # check for start checkpoint
        #if (
        #    fairseq_hydra_config.data.get("checkpoint", {}).get("restore_file", None) is not None
        #    and fairseq_hydra_config.data["checkpoint"]["restore_file"] != "checkpoint_last.pt"
        #    and os.path.exists(os.path.join(self.output_path("checkpoints", directory=True), "checkpoint_last.pt"))
        #):
        #    # start_checkpoint provided but checkpoint_last.pt exists: start_checkpoint will be ignored
        #    print(
        #        "Warning: start_checkpoint will be ignored as checkpoint_last.pt exists in output directory"
        #    )
        #    fairseq_hydra_config.data["checkpoint"]["restore_file"] = "checkpoint_last.pt"

        
        self.start_checkpoint = None
        if fairseq_hydra_config.data.get("checkpoint", {}).get("restore_file") is not None:
            self.start_checkpoint = fairseq_hydra_config.data["checkpoint"]["restore_file"]
            fairseq_hydra_config.data["checkpoint"]["restore_file"] = "checkpoint_last.pt"

        self.command_line_args = command_line_args or []
        stored_epochs = list(range(save_interval, max_epoch, save_interval)) + [max_epoch]

        self.keep_epochs = set(stored_epochs) if keep_epochs is None else set(keep_epochs)
        self.fairseq_python_exe = fairseq_python_exe if fairseq_python_exe is not None else tk.Path("/usr/bin/python3")
        self.fairseq_root = fairseq_root
        assert self.fairseq_root is not None
        self.cache_manager = cache_manager
        if isinstance(zipped_audio_dir, tk.Path):
            self.zipped_audio_dir = [zipped_audio_dir]
        else:
            self.zipped_audio_dir = zipped_audio_dir
        if self.zipped_audio_dir is not None:
            assert self.cache_manager is not CacheManagerType.none, "cache manager must be used for zipped audio input"

        self.fairseq_hydra_config = FairseqHydraTrainingJob.create_fairseq_hydra_config(**kwargs)
        # Outputs:
        self.out_fairseq_hydra_yaml = self.output_path("fairseq_hydra_config.yaml")
        self.out_checkpoint_dir = self.output_path("checkpoints", directory=True)
        self.out_models = {
            k: PytorchHydraModel(
                self.out_fairseq_hydra_yaml,
                self.output_path("checkpoints/checkpoint{}.pt".format(k)),
                k,
            )
            for k in self.keep_epochs
        }
        assert isinstance(cache_manager, CacheManagerType), "cache_manager must be instance of CacheManagerType"
        if cache_manager is not CacheManagerType.none:
            self.out_cached_audio_manifest = self.output_path("cached_audio_manifest", directory=True)
        self.out_plot_se = self.output_path("loss_and_accuracy.svg")
        self.out_plot_lr = self.output_path("learning_rate.svg")

        # Requirements:
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 4,
            "time": 4,
        }
        self.rqmt.update(rqmt or {})
        self.gpu_rqmt = self.rqmt["gpu"]
        if self.gpu_rqmt > 1:
            self.rqmt["cpu"] *= self.gpu_rqmt
            self.rqmt["mem"] *= self.gpu_rqmt

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", mini_task=True)

    @classmethod
    def create_fairseq_hydra_config(cls, fairseq_hydra_config, max_epoch, max_update, save_interval, **kwargs):
        res = copy.deepcopy(fairseq_hydra_config)
        config_dict = {}
        post_config_dict = {
            "optimization": {"max_epoch": max_epoch, "max_update": max_update},
            "checkpoint": {"save_interval": save_interval},
        }
        res.update(FairseqHydraConfig(config_dict, post_config_dict))
        return res
    
    def _fairseq_prepare_checkpoint(self, start_checkpoint):
        # rename the start checkpoint to checkpoint_last.pt if it is not None and checkpoint_last.pt does not exist
        if start_checkpoint is None:
            return
        if not os.path.exists(start_checkpoint):
            raise FileNotFoundError(f"Start checkpoint {start_checkpoint} does not exist")
        if not os.path.exists(os.path.join(self.out_checkpoint_dir.get_path(), "checkpoint_last.pt")):
            os.link(
                start_checkpoint,
                os.path.join(self.out_checkpoint_dir.get_path(), "checkpoint_last.pt")
            )
            os.link(
                start_checkpoint,
                os.path.join(self.out_checkpoint_dir.get_path(), os.path.basename(start_checkpoint))
            )

    def create_files(self):
        self.fairseq_hydra_config.write(self.out_fairseq_hydra_yaml.get_path())
        self._fairseq_prepare_checkpoint(self.start_checkpoint)
        util.create_executable("fairseq.sh", self._get_run_cmd())

    def run(self):
        if self.cache_manager is not CacheManagerType.none:
            manifest_path = self.fairseq_hydra_config.config_dict["task"]["data"].get_path()
            if self.zipped_audio_dir is None:
                for name in ["train.tsv", "valid.tsv"]:
                    with open(os.path.join(manifest_path, name), "r") as manifest_file:
                        manifest_lines = manifest_file.read().splitlines()
                    audio_path = manifest_lines[0]
                    bundle_lines = map(
                        lambda line: audio_path + "/" + line.split("\t")[0],
                        manifest_lines[1:],
                    )
                    # use i6-specific cache manager
                    if self.cache_manager is CacheManagerType.i6:
                        with open(f"{name}.bundle", "w") as bundle_file:
                            bundle_file.write("\n".join(bundle_lines))
                        try:
                            cached_audio_fn = sp.check_output(["cf", f"{name}.bundle"]).strip().decode("utf8")
                        except sp.CalledProcessError:
                            print(f"Cache manager: Error occurred for files in {name}")
                            raise
                        with open(cached_audio_fn) as local_bundle:
                            cached_bundle_lines = list(local_bundle.readlines())

                    # use general manager through gs.file_caching
                    elif self.cache_manager is CacheManagerType.general:
                        cached_bundle_lines = [gs.file_caching(l) for l in bundle_lines]

                    manifest_lines[0] = os.path.commonpath(cached_bundle_lines)
                    manifest_lines[1:] = map(
                        lambda line: os.path.relpath(line[0], manifest_lines[0]) + "\t" + line[1].split("\t")[1],
                        list(zip(cached_bundle_lines, manifest_lines[1:])),
                    )

                    with open(
                        os.path.join(self.out_cached_audio_manifest.get_path(), name),
                        "w+",
                    ) as cached_audio_manifest_file:
                        cached_audio_manifest_file.write("\n".join(manifest_lines))
            else:  # zipped audio data is given and we cache and unzip the zip file(s) instead
                local_unzipped_dir = []
                for zip_dir in self.zipped_audio_dir:
                    try:
                        # use i6-specific cache manager
                        if self.cache_manager is CacheManagerType.i6:
                            cached_audio_zip_dir = sp.check_output(["cf", zip_dir]).strip().decode("utf8")
                        # use general manager through gs.file_caching
                        elif self.cache_manager is CacheManagerType.general:
                            cached_audio_zip_dir = gs.file_caching(zip_dir.get_path()).strip()

                        local_unzipped_dir.append(os.path.join(os.path.dirname(cached_audio_zip_dir), "audio"))
                        sp.check_call(
                            [
                                "unzip",
                                "-q",
                                "-n",
                                "-j",
                                cached_audio_zip_dir,
                                "-d",
                                local_unzipped_dir[-1],
                            ]
                        )
                    except sp.CalledProcessError:
                        print(
                            f"Cache manager: Error occurred for caching and unzipping audio data in {local_unzipped_dir[-1]}"
                        )
                        raise
                common_audio_dir = os.path.commonpath(local_unzipped_dir)
                for name in ["train.tsv", "valid.tsv"]:
                    with open(os.path.join(manifest_path, name), "r") as manifest_file:
                        manifest_lines = manifest_file.read().splitlines()
                    for i in range(1, len(manifest_lines)):
                        to_check = os.path.join(common_audio_dir, manifest_lines[i].split()[0])
                        assert os.path.exists(to_check), f"Manifest file {to_check} not found in unzipped directory"
                    manifest_lines[0] = common_audio_dir
                    with open(
                        os.path.join(self.out_cached_audio_manifest.get_path(), name),
                        "w",
                    ) as cached_audio_manifest_file:
                        cached_audio_manifest_file.write("\n".join(manifest_lines))

            # symlink to other files
            for file in os.listdir(manifest_path):
                if file not in ["train.tsv", "valid.tsv"]:
                    pathlib.Path(self.out_cached_audio_manifest.get_path(), file).symlink_to(
                        pathlib.Path(manifest_path, file)
                    )
        my_env = os.environ
        if self.fairseq_root is not None:
            my_env["PYTHONPATH"] = ":".join([self.fairseq_root.get_path()] + my_env.get("PYTHONPATH", "").split(":"))
        sp.check_call(self._get_run_cmd(), env=my_env)

    def plot(self):
        directory = "./outputs"
        train_loss, train_accuracy = defaultdict(dict), defaultdict(dict)
        valid_loss, valid_accuracy = defaultdict(dict), defaultdict(dict)
        learning_rates = {}

        for cur in os.walk(directory):
            dir_path = cur[0]
            files = cur[2]
            if "hydra_train.log" in files:
                with open(f"{dir_path}/hydra_train.log", "r") as f:
                    lines = f.readlines()
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        if "[train][INFO]" in line or "[valid][INFO]" in line:
                            epoch_dict = eval(line[line.index("{") :])
                            try:
                                epoch = int(epoch_dict["epoch"])
                                losses = {k: {epoch: float(v)} for k, v in epoch_dict.items() if k.endswith("_loss")}
                                accuracy = {
                                    k: {epoch: float(v)} for k, v in epoch_dict.items() if k.endswith("_accuracy")
                                }
                            except ValueError:
                                continue
                            if "train_lr" in epoch_dict:
                                learning_rates[epoch] = float(epoch_dict["train_lr"])
                            if "[valid][INFO]" in line:
                                for k in losses.keys():
                                    valid_loss[k].update(losses[k])
                                for k in accuracy.keys():
                                    valid_accuracy[k].update(accuracy[k])
                            else:  # [train][INFO] is in line
                                for k in losses.keys():
                                    train_loss[k].update(losses[k])
                                for k in accuracy.keys():
                                    train_accuracy[k].update(accuracy[k])
                        i += 1

        colors = [
            "#2a4d6e",
            "#aa3c39",
            "#11aa00",
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, sharex=True, figsize=(12, 9))

        color_index = 0
        for k in train_loss.keys():
            axs[0].plot(
                train_loss[k].keys(),
                train_loss[k].values(),
                "o-",
                color=colors[color_index % len(colors)],
                label=k,
            )
            color_index += 1

        for k in valid_loss.keys():
            axs[0].plot(
                valid_loss[k].keys(),
                valid_loss[k].values(),
                "o-",
                color=colors[color_index % len(colors)],
                label=k,
            )
            color_index += 1

        axs[0].set_ylabel("loss")
        axs[0].legend()

        color_index = 0
        for k in train_accuracy.keys():
            axs[1].plot(
                train_accuracy[k].keys(),
                train_accuracy[k].values(),
                "o-",
                color=colors[color_index % len(colors)],
                label=k,
            )
            color_index += 1

        for k in valid_accuracy.keys():
            axs[1].plot(
                valid_accuracy[k].keys(),
                valid_accuracy[k].values(),
                "o-",
                color=colors[color_index % len(colors)],
                label=k,
            )
            color_index += 1
        axs[1].set_ylabel("accuracy")
        axs[1].set_xlabel("epochs")
        axs[1].legend()
        plt.savefig(self.out_plot_se)

        fig, axs = plt.subplots()
        axs.plot(
            learning_rates.keys(),
            learning_rates.values(),
            "o-",
            color=colors[0],
            label="learning rate",
        )
        axs.set_ylabel("learning rate")
        axs.set_xlabel("epochs")
        axs.legend()

        plt.savefig(self.out_plot_lr)

    def _get_run_cmd(self):
        run_cmd = [
            "--config-dir",
            os.path.dirname(self.out_fairseq_hydra_yaml.get_path()),
            "--config-name",
            os.path.basename(self.out_fairseq_hydra_yaml.get_path()),
        ]
        run_cmd += self.command_line_args
        run_cmd += ["checkpoint.save_dir=" + self.out_checkpoint_dir.get_path()]

        if self.cache_manager is not CacheManagerType.none:
            run_cmd += ["task.data=" + self.out_cached_audio_manifest.get_path()]

        run_cmd.insert(0, os.path.join(self.fairseq_root.get_path(), "fairseq_cli", "hydra_train.py"))

        if self.fairseq_python_exe is not None:
            run_cmd.insert(0, self.fairseq_python_exe.get_path())
        return run_cmd

    @classmethod
    def hash(cls, kwargs):
        d = {
            "command_line_args": kwargs["command_line_args"],
            "fairseq_hydra_config": FairseqHydraTrainingJob.create_fairseq_hydra_config(**kwargs),
            "fairseq_python_exe": kwargs["fairseq_python_exe"],
            "fairseq_root": kwargs["fairseq_root"],
        }
        return super().hash(d)
