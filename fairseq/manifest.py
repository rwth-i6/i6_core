from collections import defaultdict
import soundfile
import glob
import os
import random
import numpy as np
import shutil

from sisyphus import Job, Task, tk


class MergeAudioDirsJob(Job):
    """
    Merges the audio files of the same language from different folders together into one folder
    """

    def __init__(
        self, audio_dir_path_list, file_extension="wav", path_must_contain=None
    ):
        """
        :param [tk.Path] audio_dir_path_list: List of paths to folder(s) containing raw audio files
        :param str file_extension: File extension to look for in audio_dir_path
        :param str|None path_must_contain: if set, only the audio files whose path contain this substring would be included
        """
        self.audio_dir_path_list = audio_dir_path_list
        self.file_extension = file_extension
        self.path_must_contain = path_must_contain

        self.out_audio_dir_path = self.output_path("audio", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        for dir_path in self.audio_dir_path_list:
            search_path = os.path.join(dir_path, "*." + self.file_extension)
            for file in glob.iglob(search_path, recursive=True):
                if self.path_must_contain and self.path_must_contain not in file:
                    continue
                base_name = os.path.basename(file)
                dst = os.path.join(self.out_audio_dir_path, base_name)
                os.symlink(file, dst)


class CreateManifestJob(Job):
    """
    Create the manifest tsv file given the list of paths to audio directory, notice that the audio files from the same
    language should put into one directory, MergeAudioDirsJob can be used to merge the directories
    the tsv file would have content like following
    common_path
    relative_path_1 [tab] num_frames
    relative_path_1 [tab] num_frames
    """

    def __init__(self, audio_dir_path_list):
        """
        :param [tk.Path] audio_dir_path_list: List of paths to folder(s) containing raw audio files, the audio files
        from the same language should be in one directory
        """
        self.audio_dir_path_list = audio_dir_path_list
        self.common_dir = os.path.commonpath(self.audio_dir_path_list)
        self.concurrent = len(audio_dir_path_list)

        self.out_tsv_file = self.output_path("data.tsv")

        self.rqmt = {"time": 2, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge", mini_task=True)

    def run(self, task_id):
        with open("%d.tsv" % task_id, "w") as f:
            for path in glob.iglob(
                self.audio_dir_path_list[task_id - 1].get_path() + "/*", recursive=True
            ):
                frames = soundfile.info(path).frames
                rel_path = os.path.relpath(path, self.common_dir)
                print("{}\t{}".format(rel_path, frames), file=f)

    def merge(self):
        with open(self.out_tsv_file, "w") as f_out:
            print(self.common_dir, file=f_out)
            for idx in range(1, self.concurrent + 1):
                with open("%d.tsv" % idx) as f_in:
                    shutil.copyfileobj(f_in, f_out, length=16 * 1024)


class SplitTrainCvDataJob(Job):
    """
    This job splits the train and cv dataset based on the given portion
    """

    def __init__(self, tsv_file, valid_portion, seed=42):
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
        ) as f_train, open(
            os.path.join(self.out_manifest_path, "valid.tsv"), "w"
        ) as f_valid:
            while True:
                line = f_in.readline()
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
                if not line:
                    break


class BalanceMultiLingualDatatJob(Job):
    """
    Balance the multilingual data by up-sampling the low-resource langauge.
    Up-sampling is implemented by duplicating the input audio files of the low-resource langauge.
    The up-sampling factor is computed according to the paper "Unsupervised Cross-Lingual Representation Learning for
    Speech Recognition", see  arXiv:2006.13979v2
    """

    def __init__(self, train_tsv_file, alpha):
        """
        :param [tk.Path] train_tsv_file: the tsv file of the train dataset which needs to be balanced
        :param float alpha: the parameter that controls the importance given to high-resource versus
        low-resource languages during pretraining. The lower the parameter value, the higher the up-sampling factor
        would be given for the low-resource langauge
        """
        self.train_tsv_file = train_tsv_file
        self.alpha = alpha

        self.out_tsv_file = self.output_path("train.tsv")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        assert self.alpha is not None and self.alpha < 1
        all_train_data = defaultdict(list)
        with open(self.train_tsv_file, "r") as f_in, open(
            self.out_tsv_file, "w"
        ) as f_out:
            while True:
                line = f_in.readline()
                if "\t" not in line:
                    f_out.write(line)
                else:
                    rel_path, frames = line.strip().split("\t")
                    rel_path_dir = os.path.dirname(rel_path)
                    all_train_data[rel_path_dir].append((rel_path, int(frames)))
                if not line:
                    break

            num_corpora = len(all_train_data.keys())
            corpora_lengths = [
                sum([frames for _, frames in train_data])
                for train_data in all_train_data.values()
            ]
            sum_corpora_length = sum(corpora_lengths)

            corpora_probs = [(l / sum_corpora_length) for l in corpora_lengths]
            upsampling_proportions = [p**self.alpha for p in corpora_probs]
            upsampling_factors = [
                upsampling_proportions[i] / corpora_probs[i] for i in range(num_corpora)
            ]
            upsampling_factors = [
                f / min(upsampling_factors) for f in upsampling_factors
            ]

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
                        upsampled_train_data[i].append(
                            list(all_train_data.values())[i][j]
                        )
                        added_length += list(all_train_data.values())[i][j][1]
                        j += 1

            upsampled_train_data_lengths = []
            for train_data in upsampled_train_data:
                upsampled_train_data_lengths.append(
                    sum([frames for _, frames in train_data])
                )
            actual_upsampled_probabilities = np.array(
                upsampled_train_data_lengths
            ) / sum(upsampled_train_data_lengths)
            # following test might fail if the input corpora are tiny
            upsampled_probabilities = [
                p / sum(upsampling_proportions) for p in upsampling_proportions
            ]
            np.testing.assert_allclose(
                upsampled_probabilities, actual_upsampled_probabilities, rtol=0.1
            )

            for i, train_data in enumerate(upsampled_train_data):
                for path, frames in train_data:
                    print("{}\t{}".format(path, frames), file=f_out)


class FairseqAudioManifestCreationJob(Job):
    """
    Creates required manifest files for wav2vec pretraining with fairseq. For the
    script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py
    """

    __sis_hash_exclude__ = {
        "alpha": None,
        "manifest_audio_paths": None,
    }

    def __init__(
        self,
        audio_dir_paths,
        file_extension="wav",
        valid_portion=0.01,
        seed=42,
        path_must_contain=None,
        alpha=None,
        manifest_audio_paths=None,
    ):
        """
        :param [tk.Path]|tk.Path audio_dir_paths: List of paths or single path to folder(s) containing raw audio files to be included
        :param str file_extension: File extension to look for in audio_dir_path
        :param float valid_portion: portion of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        :param str|None path_must_contain: if set, path must contain this substring
            for a file to be included in the manifest
        :param float alpha: Specifies how much to upsample low resource languages. Upsampling
            calculation is done according to "Unsupervised Cross-Lingual Representation Learning for Speech
            Recognition", see  arXiv:2006.13979v2
        :param [tk.Path]|tk.Path|None manifest_audio_paths: Explicitly specifies output paths in manifest for each
            audio directory respectively. Allows to use different paths in the manifest than in the audio_dir_paths
        """
        if isinstance(audio_dir_paths, tk.Path):
            self.audio_dir_paths = [audio_dir_paths]
        else:
            assert isinstance(audio_dir_paths, list)
            self.audio_dir_paths = audio_dir_paths
        assert all([isinstance(path, tk.Path) for path in self.audio_dir_paths])

        if len(self.audio_dir_paths) == 1:
            assert (
                alpha is None
            ), "Only one audio directory is given, upsampling not possible"

        if manifest_audio_paths:
            if isinstance(manifest_audio_paths, tk.Path):
                manifest_audio_paths = [manifest_audio_paths]
            assert len(manifest_audio_paths) == len(self.audio_dir_paths)
            self.manifest_audio_paths = [
                os.path.realpath(path) for path in manifest_audio_paths
            ]
        else:
            self.manifest_audio_paths = None

        self.file_extension = file_extension
        self.valid_portion = valid_portion
        assert 0.0 <= self.valid_portion <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain
        self.alpha = alpha
        if self.alpha is not None:
            assert 0.0 <= self.alpha <= 1.0

        self.out_manifest_path = self.output_path("manifest", directory=True)
        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        rand = random.Random(self.seed)

        dir_paths = [os.path.realpath(path.get_path()) for path in self.audio_dir_paths]

        if self.manifest_audio_paths:
            common_dir = os.path.commonpath(self.manifest_audio_paths)
        else:
            common_dir = os.path.commonpath(dir_paths)

        all_train_data = []

        valid_f = (
            open(os.path.join(self.out_manifest_path, "valid.tsv"), "w")
            if self.valid_portion > 0
            else None
        )
        if valid_f is not None:
            print(common_dir, file=valid_f)

        for i, dir_path in enumerate(dir_paths):
            train_data = []
            search_path = os.path.join(dir_path, "*." + self.file_extension)
            for path in glob.iglob(search_path, recursive=True):
                frames = soundfile.info(path).frames
                path = os.path.realpath(path)

                if self.path_must_contain and self.path_must_contain not in path:
                    continue

                if self.manifest_audio_paths:
                    rel_path = os.path.relpath(self.manifest_audio_paths[i], common_dir)
                    rel_path = os.path.join(rel_path, os.path.basename(path))
                else:
                    rel_path = os.path.relpath(path, common_dir)
                if rand.random() > self.valid_portion:
                    train_data.append((rel_path, frames))
                else:
                    print("{}\t{}".format(rel_path, frames), file=valid_f)
            all_train_data.append(train_data)
        if valid_f is not None:
            valid_f.close()

        if self.alpha is not None and self.alpha < 1:
            corpora_lengths = []
            for train_data in all_train_data:
                corpora_lengths.append(sum([frames for _, frames in train_data]))

            num_corpora = len(corpora_lengths)
            sum_corpora_length = sum(corpora_lengths)
            corpora_probs = [(l / sum_corpora_length) for l in corpora_lengths]
            upsampling_proportions = [p**self.alpha for p in corpora_probs]
            upsampling_factors = [
                upsampling_proportions[i] / corpora_probs[i] for i in range(num_corpora)
            ]
            upsampling_factors = [
                f / min(upsampling_factors) for f in upsampling_factors
            ]

            # assert all entries larger than 1 because we only want to
            # upsample and never downsample
            assert min(upsampling_factors >= 1)

            upsampled_train_data = []

            for i, upsample in enumerate(upsampling_factors):
                upsampled_train_data.append([])
                while upsample >= 1:
                    upsampled_train_data[i].extend(all_train_data[i])
                    upsample -= 1
                if upsample > 0:
                    added_length = 0
                    random.shuffle(all_train_data[i])
                    j = 0
                    while corpora_lengths[i] * upsample > added_length:
                        upsampled_train_data[i].append(all_train_data[i][j])
                        added_length += all_train_data[i][j][1]
                        j += 1

            upsampled_train_data_lengths = []
            for train_data in upsampled_train_data:
                upsampled_train_data_lengths.append(
                    sum([frames for _, frames in train_data])
                )
            actual_upsampled_probabilities = np.array(
                upsampled_train_data_lengths
            ) / sum(upsampled_train_data_lengths)
            # following test might fail if the input corpora are really small
            upsampled_probabilities = [
                p / sum(upsampling_proportions) for p in upsampling_proportions
            ]
            np.testing.assert_allclose(
                upsampled_probabilities, actual_upsampled_probabilities, rtol=0.1
            )
        else:
            upsampled_train_data = all_train_data

        with open(os.path.join(self.out_manifest_path, "train.tsv"), "w") as train_f:
            print(common_dir, file=train_f)
            for i, train_data in enumerate(upsampled_train_data):
                for path, frames in train_data:
                    print("{}\t{}".format(path, frames), file=train_f)
