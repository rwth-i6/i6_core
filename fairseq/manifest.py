from collections import defaultdict
import errno
import glob
import numpy as np
import os
import random
import shutil
import soundfile
from typing import Sequence, Optional, Type

from sisyphus import Job, Task, tk


class MergeAudioDirsJob(Job):
    """
    Merges the audio files of the same language from different folders together into one folder
    """

    def __init__(
        self,
        audio_dir_paths: Sequence[Type[tk.Path]],
        file_extension: str = "wav",
        path_must_contain: Optional[str] = None,
    ):
        """
        :param [tk.Path] audio_dir_paths: list of paths to folder(s) containing raw audio files
        :param str file_extension: file extension to look for in audio_dir_path
        :param str|None path_must_contain: if set, only the audio files whose path contain this substring would be included
        """
        self.audio_dir_paths = audio_dir_paths
        self.file_extension = file_extension
        self.path_must_contain = path_must_contain
        self.concurrent = len(audio_dir_paths)

        self.out_audio_dir_path = self.output_path("audio", directory=True)

        self.rqmt = {"time": 8, "cpu": 4, "mem": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        dir_path = self.audio_dir_paths[task_id - 1]
        search_path = os.path.join(dir_path, "*." + self.file_extension)
        for file in glob.iglob(search_path, recursive=True):
            if self.path_must_contain and self.path_must_contain not in file:
                continue
            base_name = os.path.basename(file)
            creation_complete = False
            dst = os.path.join(self.out_audio_dir_path, base_name)
            i = 2
            while not creation_complete:
                while os.path.exists(dst):
                    if os.path.realpath(dst) == file:
                        creation_complete = True
                        break
                    dst = f"{os.path.splitext(dst)[0]}_{i}.{self.file_extension}"
                    i += 1
                else:
                    try:
                        os.symlink(file, dst)
                        creation_complete = True
                    except OSError as err:
                        if err.errno != errno.EEXIST:
                            raise err


class CreateManifestJob(Job):
    """
    Create the manifest tsv file given the list of paths to audio directory, notice that the audio files from the same
    language should put into one directory, MergeAudioDirsJob can be used to merge the directories
    the tsv file would have content like following
    common_path
    relative_path_1 [tab] num_frames
    relative_path_1 [tab] num_frames
    """

    def __init__(self, audio_dir_paths: Sequence[Type[tk.Path]]):
        """
        :param [tk.Path] audio_dir_paths: List of paths to folder(s) containing raw audio files, the audio files
        from the same language should be in one directory
        """
        self.audio_dir_paths = audio_dir_paths
        self.common_dir = os.path.commonpath(self.audio_dir_paths)
        self.concurrent = len(audio_dir_paths)

        self.out_tsv_file = self.output_path("data.tsv")

        self.rqmt = {"time": 2, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge", mini_task=True)

    def run(self, task_id):
        with open("%d.tsv" % task_id, "w") as f:
            for path in glob.iglob(self.audio_dir_paths[task_id - 1].get_path() + "/*", recursive=True):
                frames = soundfile.info(path).frames
                rel_path = os.path.relpath(path, self.common_dir)
                f.write("{}\t{}\n".format(rel_path, frames))

    def merge(self):
        with open(self.out_tsv_file, "w") as f_out:
            f_out.write(self.common_dir + "\n")
            for idx in range(1, self.concurrent + 1):
                with open("%d.tsv" % idx) as f_in:
                    shutil.copyfileobj(f_in, f_out, length=16 * 1024)
                os.remove("%d.tsv" % idx)


class SplitTrainCvDataJob(Job):
    """
    This job splits the train and cv dataset based on the given portion
    """

    def __init__(self, tsv_file: Type[tk.Path], valid_portion: float, seed: float = 42):
        """
        :param [tk.Path] tsv_file: the tsv file of the dataset which needs to be split
        :param float valid_portion: portion of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        """
        self.tsv_file = tsv_file
        self.valid_portion = valid_portion
        self.seed = seed

        self.out_manifest_path = self.output_path("manifest", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rand = random.Random(self.seed)
        with open(self.tsv_file, "r") as f_in, open(
            os.path.join(self.out_manifest_path, "train.tsv"), "w"
        ) as f_train, open(os.path.join(self.out_manifest_path, "valid.tsv"), "w") as f_valid:
            for line in f_in:
                # common path
                if "\t" not in line:
                    f_train.write(line)
                    f_valid.write(line)
                else:
                    # write to train
                    if rand.random() > self.valid_portion:
                        f_train.write(line)
                    # write to valid
                    else:
                        f_valid.write(line)


class BalanceMultiLingualDatatJob(Job):
    """
    Balance the multilingual data by up-sampling the low-resource langauge.
    Up-sampling is implemented by duplicating the input audio files of the low-resource langauge.
    The up-sampling factor is computed according to the paper "Unsupervised Cross-Lingual Representation Learning for
    Speech Recognition", see  arXiv:2006.13979v2
    """

    def __init__(
        self,
        train_tsv_file: Type[tk.Path],
        alpha: float,
    ):
        """
        :param tk.Path train_tsv_file: the tsv file of the train dataset which needs to be balanced
        :param float alpha: the parameter that controls the importance given to high-resource versus
        low-resource languages during pretraining. The lower the parameter value, the higher the up-sampling factor
        would be given for the low-resource langauge
        """
        self.train_tsv_file = train_tsv_file
        assert alpha is not None and alpha < 1
        self.alpha = alpha

        self.out_tsv_file = self.output_path("train.tsv")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        all_train_data = defaultdict(list)
        with open(self.train_tsv_file, "r") as f_in, open(self.out_tsv_file, "w") as f_out:
            for line in f_in:
                if "\t" not in line:
                    f_out.write(line)
                else:
                    rel_path, frames = line.strip().split("\t")
                    rel_path_dir = os.path.dirname(rel_path)
                    all_train_data[rel_path_dir].append((rel_path, int(frames)))

            num_corpora = len(all_train_data.keys())
            corpora_lengths = [sum([frames for _, frames in train_data]) for train_data in all_train_data.values()]
            sum_corpora_length = sum(corpora_lengths)

            corpora_probs = [(l / sum_corpora_length) for l in corpora_lengths]
            upsampling_proportions = [p**self.alpha for p in corpora_probs]
            upsampling_factors = [upsampling_proportions[i] / corpora_probs[i] for i in range(num_corpora)]
            upsampling_factors = [f / min(upsampling_factors) for f in upsampling_factors]

            upsampled_train_data = []
            for i, upsample in enumerate(upsampling_factors):
                upsampled_train_data.append([])
                while upsample >= 1:
                    upsampled_train_data[i].extend(list(all_train_data.values())[i])
                    upsample -= 1
                if upsample > 0:
                    added_length = 0
                    random.shuffle(list(all_train_data.values())[i])
                    j = 0
                    while corpora_lengths[i] * upsample > added_length:
                        upsampled_train_data[i].append(list(all_train_data.values())[i][j])
                        added_length += list(all_train_data.values())[i][j][1]
                        j += 1

            upsampled_train_data_lengths = []
            for train_data in upsampled_train_data:
                upsampled_train_data_lengths.append(sum([frames for _, frames in train_data]))
            actual_upsampled_probabilities = np.array(upsampled_train_data_lengths) / sum(upsampled_train_data_lengths)
            # following test might fail if the input corpora are tiny
            upsampled_probabilities = [p / sum(upsampling_proportions) for p in upsampling_proportions]
            np.testing.assert_allclose(upsampled_probabilities, actual_upsampled_probabilities, rtol=0.1)

            for i, train_data in enumerate(upsampled_train_data):
                for path, frames in train_data:
                    f_out.write("{}\t{}\n".format(path, frames))
