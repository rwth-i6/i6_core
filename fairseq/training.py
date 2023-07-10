import logging
import os
import subprocess as sp
import yaml
import sys
import copy
import re

from sisyphus import Job, tk

import i6_core.util as util


class FairseqHydraConfig:
    """
    An object that manages a Fairseq hydra config.
    """

    def __init__(self, config_dict: Dict[str, Any], *, package_name: str = ""):
        """
        :param config_dict: Contains the information which is needed for fairseq-hydra-train. Will be converted and dumped into a .yaml
        :param package_name: The @package directory that is required to be added to the top of Hydra config, for example "# @package _group_"
        """
        assert isinstance(config_dict, dict)
        self.config_dict = config_dict
        self.package_name = package_name

    def write(self, path: str):
        path_corrected_config = self.config_dict.copy()
        # recursively go through config dictionary to get all sisyphus paths inplace
        path_corrected_config = util.instanciate_delayed(path_corrected_config)

        config_yaml = yaml.dump(path_corrected_config)
        # "# @package _group_" was written at the beginning in the example .yaml from fairseq:
        if self.package_name != "":
            config_yaml = f"# @package {self.package_name}\n" + config_yaml
        with open(path, "w") as file:
            file.write(config_yaml)


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


class FairseqHydraTrainingJob(Job):
    """
    Train a Fairseq model using fairseq-hydra-train
    """

    def __init__(
        self,
        fairseq_hydra_config,
        *,  # args below are keyword only
        command_line_args=None,
        max_epoch=None,
        save_interval=None,
        keep_epochs=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        gpu_rqmt=1,
        fairseq_python_exe=None,
        fairseq_root=None,
        use_cache_manager=True,
        zipped_audio_dir=None,
    ):
        """
        :param FairseqHydraConfig fairseq_hydra_config:
        :param list command_line_args: Additional command line arguments (starting with "--*"),
            to configure the Fairseq-hydra task
        :param int|None max_epoch: maximum number of epochs to run. Note that this value IS currently HASHED.
        :param int|None save_interval: save a checkpoint each n-th epoch
        :param list[int]|set[int]|None keep_epochs: specify which checkpoints are kept in self.out_models.
            Use None for each save_interval-th epoch
        :param int|float time_rqmt: Overall time requirements
        :param int|float mem_rqmt: Memory requirements (per GPU)
        :param int cpu_rqmt: Required number of CPUs (per GPU)
        :param int gpu_rqmt: Number of required GPUs
        :param Path fairseq_python_exe: File path to the executable for running python
        :param Path fairseq_root: File path to the fairseq git for alternative call of fairseq-hydra-train
            (no need to install fairseq here)
        :param bool use_cache_manager: enables caching of data given in the manifest with the i6 cache manager
        :param [tk.Path]|tk.Path zipped_audio_dir: using a bundle file for caching is very slow for large manifests. For
            speeding up the audio file transfer using the cache manager, a zipped audio directory might be provided.
            The zipped audio directory is then used for caching instead and unzipped on the node for training
        """

        # Inputs:
        self.fairseq_hydra_config = fairseq_hydra_config
        self.command_line_args = command_line_args or []
        warning_text = "is specified as input arg and in fairseq_hydra_config. We take the input arg"
        save_interval_config = fairseq_hydra_config.config_dict.get(
            "checkpoint", {}
        ).get("save_interval", None)
        if save_interval is not None and save_interval_config is not None:
            logging.warning(
                "'save_interval' {}: {}".format(warning_text, save_interval)
            )
        self.save_interval = save_interval or save_interval_config
        assert (
            self.save_interval is not None
        ), "save_interval has to be set explicitly or via fairseq_hydra_config"
        max_epoch_config = fairseq_hydra_config.config_dict.get("optimization", {}).get(
            "max_epoch", None
        )
        if max_epoch is not None and max_epoch_config is not None:
            logging.warning("'max_epoch' {}: {}".format(warning_text, max_epoch))
        self.max_epoch = max_epoch or max_epoch_config
        assert (
            self.max_epoch is not None
        ), "max_epoch has to be set explicitly or via fairseq_hydra_config"
        stored_epochs = list(
            range(self.save_interval, self.max_epoch, self.save_interval)
        ) + [self.max_epoch]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)
        self.fairseq_python_exe = (
            fairseq_python_exe
            if fairseq_python_exe is not None
            else getattr(gs, "FAIRSEQ_PYTHON_EXE", None)
        )
        self.fairseq_root = fairseq_root
        # We assume that only one of the two possible entry points is given as an input
        assert self.fairseq_root is not None
        if self.fairseq_root is not None:
            assert self.fairseq_python_exe is not None
        self.use_cache_manager = use_cache_manager
        if isinstance(zipped_audio_dir, tk.Path):
            self.zipped_audio_dir = [zipped_audio_dir]
        else:
            self.zipped_audio_dir = zipped_audio_dir
        if self.zipped_audio_dir is not None:
            assert (
                self.use_cache_manager
            ), "cache manager must be used for zipped audio input"

        # Outputs:
        self.out_fairseq_hydra_yaml = self.output_path("fairseq_hydra_config.yaml")
        self.out_checkpoint_dir = self.output_path("checkpoints", directory=True)
        self.out_models = {
            k: PytorchHydraModel(
                self.out_fairseq_hydra_yaml,
                self.output_path("checkpoints/checkpoint{}.pt".format(k)),
                k,
            )
            for k in stored_epochs
            if k in self.keep_epochs
        }
        self.out_cached_audio_manifest = self.output_path(
            "cached_audio_manifest", directory=True
        )
        self.out_plot_se = self.output_path("score_and_error.svg")
        self.out_plot_lr = self.output_path("learning_rate.svg")

        # Requirements:
        self.gpu_rqmt = gpu_rqmt
        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }
        if self.gpu_rqmt > 1:
            self.rqmt["cpu"] *= self.gpu_rqmt
            self.rqmt["mem"] *= self.gpu_rqmt

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", mini_task=True)

    def create_files(self):
        self.fairseq_hydra_config.write(self.out_fairseq_hydra_yaml.get_path())
        util.create_executable("fairseq.sh", self._get_run_cmd())

    def run(self):
        if self.use_cache_manager:
            manifest_path = self.fairseq_hydra_config.config_dict["task"][
                "data"
            ].get_path()
            if self.zipped_audio_dir is None:
                for name in ["train.tsv", "valid.tsv"]:
                    with open(os.path.join(manifest_path, name), "r") as manifest_file:
                        manifest_lines = manifest_file.read().splitlines()
                    audio_path = manifest_lines[0]
                    bundle_lines = map(
                        lambda line: audio_path + "/" + line.split("\t")[0],
                        manifest_lines[1:],
                    )
                    with open(f"{name}.bundle", "w") as bundle_file:
                        bundle_file.write("\n".join(bundle_lines))
                    try:
                        cached_audio_fn = (
                            sp.check_output(["cf", f"{name}.bundle"])
                            .strip()
                            .decode("utf8")
                        )
                    except sp.CalledProcessError:
                        print(f"Cache manager: Error occurred for files in {name}")
                        raise

                    with open(cached_audio_fn) as local_bundle:
                        bundle_lines = list(local_bundle.readlines())
                        manifest_lines[0] = os.path.commonpath(bundle_lines)
                        manifest_lines[1:] = map(
                            lambda line: os.path.relpath(line, manifest_lines[0]),
                            manifest_lines[1:],
                        )
                    with open(
                        os.path.join(self.out_cached_audio_manifest.get_path(), name),
                        "w",
                    ) as cached_audio_manifest_file:
                        cached_audio_manifest_file.write("\n".join(manifest_lines))
            else:  # zipped audio data is given and we cache and unzip the zip file(s) instead
                local_unzipped_dir = []
                for zip_dir in self.zipped_audio_dir:
                    try:
                        cached_audio_zip_dir = (
                            sp.check_output(["cf", zip_dir]).strip().decode("utf8")
                        )
                        local_unzipped_dir.append(
                            os.path.join(os.path.dirname(cached_audio_zip_dir), "audio")
                        )
                        sp.check_call(
                            [
                                "unzip",
                                "-q",
                                "-n",
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
                        to_check = os.path.join(
                            common_audio_dir, manifest_lines[i].split()[0]
                        )
                        assert os.path.exists(
                            to_check
                        ), f"Manifest file {to_check} not found in unzipped directory"
                    manifest_lines[0] = common_audio_dir
                    with open(
                        os.path.join(self.out_cached_audio_manifest.get_path(), name),
                        "w",
                    ) as cached_audio_manifest_file:
                        cached_audio_manifest_file.write("\n".join(manifest_lines))

        my_env = os.environ
        if self.fairseq_root is not None:
            my_env["PYTHONPATH"] = ":".join(
                [self.fairseq_root] + my_env.get("PYTHONPATH", "").split(":")
            )
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
                    for i in range(len(lines) - 1):
                        line = lines[i]
                        if "begin validation on" in line or "end of epoch" in line:
                            epoch_dict = eval(lines[i + 1][lines[i + 1].index("{") :])
                            try:
                                epoch = int(epoch_dict["epoch"])
                                losses = {
                                    k: {epoch: float(v)}
                                    for k, v in epoch_dict.items()
                                    if k.endswith("_loss")
                                }
                                accuracy = {
                                    k: {epoch: float(v)}
                                    for k, v in epoch_dict.items()
                                    if k.endswith("_accuracy")
                                }
                            except ValueError:
                                continue
                            if "train_lr" in epoch_dict:
                                learning_rates[epoch] = float(epoch_dict["train_lr"])
                            if "begin validation on" in line:
                                for k in losses.keys():
                                    valid_loss[k].update(losses[k])
                                for k in accuracy.keys():
                                    valid_accuracy[k].update(accuracy[k])
                            else:
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
                color=colors[color_index],
                label=k,
            )
            color_index += 1

        for k in valid_loss.keys():
            axs[0].plot(
                valid_loss[k].keys(),
                valid_loss[k].values(),
                "o-",
                color=colors[color_index],
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
                color=colors[color_index],
                label=k,
            )
            color_index += 1

        for k in valid_accuracy.keys():
            axs[1].plot(
                valid_accuracy[k].keys(),
                valid_accuracy[k].values(),
                "o-",
                color=colors[color_index],
                label=k,
            )
            color_index += 1
        axs[1].set_ylabel("accuracy")
        axs[1].set_xlabel("epochs")
        axs[1].legend()
        plt.savefig("score_error.svg")

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

        plt.savefig("learning_rate.svg")

    def _get_run_cmd(self):
        run_cmd = [
            "--config-dir",
            os.path.dirname(self.out_fairseq_hydra_yaml.get_path()),
            "--config-name",
            os.path.basename(self.out_fairseq_hydra_yaml.get_path()),
        ]
        run_cmd += self.command_line_args
        run_cmd += ["checkpoint.save_dir=" + self.out_checkpoint_dir.get_path()]
        if self.save_interval != self.fairseq_hydra_config.config_dict.get(
            "checkpoint", {}
        ).get("save_interval", None):
            run_cmd += ["checkpoint.save_interval=" + str(self.save_interval)]
        if self.max_epoch != self.fairseq_hydra_config.config_dict.get(
            "optimization", {}
        ).get("max_epoch", None):
            run_cmd += ["optimization.max_epoch=" + str(self.max_epoch)]

        if self.use_cache_manager:
            run_cmd += ["task.data=" + self.out_cached_audio_manifest.get_path()]

        sys.path.insert(0, self.fairseq_root)
        hydra_train_entry = self.fairseq_root + "fairseq_cli/hydra_train.py"
        run_cmd.insert(0, tk.uncached_path(hydra_train_entry))

        if self.fairseq_python_exe is not None:
            run_cmd.insert(0, tk.uncached_path(self.fairseq_python_exe))
        return run_cmd

    @classmethod
    def hash(cls, kwargs):
        d = copy.copy(kwargs)
        d.pop("use_cache_manager", None)
        d.pop("zipped_audio_dir", None)
        d.pop("time_rqmt", None)
        d.pop("mem_rqmt", None)
        d.pop("cpu_rqmt", None)
        d.pop("gpu_rqmt", None)
        return super().hash(d)
