import re
from typing import Dict, Any
import os
from tqdm import tqdm

from sisyphus import Job, Task, Path

from i6_core.lib.corpus import Corpus, Recording, Segment

import subprocess as sp
from io import BytesIO

filters = {
    "voxpopuli": re.compile("PLENARY"),
    "commonvoice": re.compile("common_voice"),
    "librispeech": re.compile("^[0-9-]*$"),
    "yodas": re.compile(".wav$"),
}


def extract_audio_from_text(entry, output_path):
    sp.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libvorbis",
            "-b:a",
            "16k",
            os.path.join(output_path, entry["ID"] + ".ogg"),
        ],
        input=BytesIO(entry["wav"]["bytes"]).read(),
        check=True,
    )
    entry.pop("wav")
    return entry


def remove_audio_from_entry(entry):
    entry.pop("wav")
    return entry


def create_recording_from_entry(entry, out_dir: str):
    recording = Recording()
    recording.name = entry["ID"]
    recording.audio = os.path.join(out_dir, entry["ID"] + ".ogg")

    segment = Segment()
    segment.name = "0"
    segment.start = 0.0
    segment.end = entry["duration"]
    segment.orth = entry["text"]

    recording.add_segment(segment)
    return recording


class PrepareLoquaciousDatasetJob(Job):
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        return super().hash(d)


class PrepareLoquaciousTrainSmallDatasetJob(PrepareLoquaciousDatasetJob):
    """
    Prepare the Loquacious dataset from HuggingFace.
    """

    def __init__(
        self,
        hf_home_dir: Path,
    ):
        self.hf_home_dir = hf_home_dir

        self.out_corpus = self.output_path("loquacious_train_small.xml.gz")
        self.out_dir = self.output_path("audio", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 16, "mem": 32, "time": 4})

    def run(self):
        os.environ["HF_HOME"] = self.hf_home_dir.get_path()
        from datasets import load_dataset

        dataset = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="small",
            split="train",
            num_proc=8,
        )
        print("extract audio")
        text_only_dataset = dataset.map(
            extract_audio_from_text,
            fn_kwargs={"output_path": self.out_dir.get_path()},
            num_proc=8,
            load_from_cache_file=False,
        )
        print("start corpus creation")
        corpus = Corpus()
        corpus.name = "loquacious-train-small"
        progress = tqdm(text_only_dataset, file=open("/dev/null", "w"))
        for i, entry in enumerate(progress):
            if i % 100 == 0:
                # do not bloat log files, so print manually
                print(progress)
            recording = create_recording_from_entry(entry, self.out_dir.get_path())
            corpus.add_recording(recording)

        corpus.dump(self.out_corpus.get_path())


class PrepareLoquaciousTrainMediumDatasetJob(PrepareLoquaciousDatasetJob):
    """
    Prepare the Loquacious dataset from HuggingFace.
    """

    def __init__(
        self,
        hf_home_dir: Path,
    ):
        self.hf_home_dir = hf_home_dir

        self.out_corpus = self.output_path("loquacious_train_medium.xml.gz")
        self.out_corpus_wo_small = self.output_path("loquacious_train_medium_wo_small.xml.gz")
        self.out_dir = self.output_path("audio", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 8, "mem": 32, "time": 40})

    def run(self):
        os.environ["HF_HOME"] = self.hf_home_dir.get_path()
        from datasets import load_dataset

        print("load small and medium datasets using HF")
        dataset_medium = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="medium",
            split="train",
            num_proc=8,
        )
        dataset_small = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="small",
            split="train",
            num_proc=8,
        )

        print("remove audio from small set")
        text_only_dataset_small = dataset_small.map(remove_audio_from_entry, num_proc=8, load_from_cache_file=False)

        print("extract audio from medium set")
        text_only_dataset_medium = dataset_medium.map(
            extract_audio_from_text,
            fn_kwargs={"output_path": self.out_dir.get_path()},
            num_proc=8,
            load_from_cache_file=False,
        )

        print("get sequence IDs of small set")
        small_ids = set()
        progress = tqdm(text_only_dataset_small, file=("/dev/null", "w"))
        for i, entry in enumerate(progress):
            if i % 100 == 0:
                print(progress)
            small_ids.add(entry["ID"])

        print("start corpus creation")
        corpus = Corpus()
        corpus.name = "loquacious-train-medium"
        corpus_wo_small = Corpus()
        corpus_wo_small.name = "loquacious-train-medium-wo-small"
        progress = tqdm(text_only_dataset_medium, file=open("/dev/null", "w"))
        for i, entry in enumerate(progress):
            if i % 1000 == 0:
                print(progress)
            recording = create_recording_from_entry(entry, self.out_dir.get_path())
            corpus.add_recording(recording)
            if entry["ID"] not in small_ids:
                corpus_wo_small.add_recording(recording)

        corpus.dump(self.out_corpus.get_path())
        corpus_wo_small.dump(self.out_corpus_wo_small.get_path())


class PrepareLoquaciousTestDatasetsJob(PrepareLoquaciousDatasetJob):
    """
    Prepare the Loquacious dataset from HuggingFace.
    """

    def __init__(
        self,
        hf_home_dir: Path,
    ):
        self.hf_home_dir = hf_home_dir

        self.out_dev_all = self.output_path("loquacious_dev_all.xml.gz")
        self.out_test_all = self.output_path("loquacious_test_all.xml.gz")
        self.out_dev_librispeech = self.output_path("loquacious_dev_librispeech.xml.gz")
        self.out_test_librispeech = self.output_path("loquacious_test_librispeech.xml.gz")
        self.out_dev_commonvoice = self.output_path("loquacious_dev_commonvoice.xml.gz")
        self.out_test_commonvoice = self.output_path("loquacious_test_commonvoice.xml.gz")
        self.out_dev_voxpopuli = self.output_path("loquacious_dev_voxpopuli.xml.gz")
        self.out_test_voxpopuli = self.output_path("loquacious_test_voxpopuli.xml.gz")
        self.out_dev_yodas = self.output_path("loquacious_dev_yodas.xml.gz")
        self.out_test_yodas = self.output_path("loquacious_test_yodas.xml.gz")

        self.out_dev_corpora = {
            "all": self.out_dev_all,
            "librispeech": self.out_dev_librispeech,
            "commonvoice": self.out_dev_commonvoice,
            "voxpopuli": self.out_dev_voxpopuli,
            "yodas": self.out_dev_yodas,
        }

        self.out_test_corpora = {
            "all": self.out_test_all,
            "librispeech": self.out_test_librispeech,
            "commonvoice": self.out_test_commonvoice,
            "voxpopuli": self.out_test_voxpopuli,
            "yodas": self.out_test_yodas,
        }

        self.out_dir = self.output_path("audio", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 16, "mem": 32, "time": 4})

    def run(self):
        os.environ["HF_HOME"] = self.hf_home_dir.get_path()
        from datasets import load_dataset

        dataset_dev = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="small",
            split="dev",
            num_proc=8,
        )
        dataset_test = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="small",
            split="test",
            num_proc=8,
        )
        print("extract audio")
        text_only_dataset_dev = dataset_dev.map(
            extract_audio_from_text,
            fn_kwargs={"output_path": self.out_dir.get_path()},
            num_proc=8,
            load_from_cache_file=False,
        )
        text_only_dataset_test = dataset_test.map(
            extract_audio_from_text,
            fn_kwargs={"output_path": self.out_dir.get_path()},
            num_proc=8,
            load_from_cache_file=False,
        )

        print("start corpus creation")
        corpus_keys = [
            "all",
            "librispeech",
            "commonvoice",
            "voxpopuli",
            "yodas",
        ]
        dev_corpora = {key: Corpus() for key in corpus_keys}
        test_corpora = {key: Corpus() for key in corpus_keys}
        for key, corpus in dev_corpora.items():
            corpus.name = f"loquacious-dev-{key}"
        for key, corpus in test_corpora.items():
            corpus.name = f"loquacious-test-{key}"

        for entry in tqdm(text_only_dataset_dev):
            recording = create_recording_from_entry(entry, self.out_dir.get_path())
            dev_corpora["all"].add_recording(recording)
            matched = False
            for key, filter in filters.items():
                if filter.search(entry["ID"]):
                    matched = True
                    dev_corpora[key].add_recording(recording)
            assert matched, f"no filter matched for {entry['ID']}"

        for entry in tqdm(text_only_dataset_test):
            recording = create_recording_from_entry(entry, self.out_dir.get_path())
            test_corpora["all"].add_recording(recording)
            matched = False
            for key, filter in filters.items():
                if filter.search(entry["ID"]):
                    matched = True
                    test_corpora[key].add_recording(recording)
            assert matched, f"no filter matched for {entry['ID']}"

        for key, dev_corpus in dev_corpora.items():
            dev_corpus.dump(self.out_dev_corpora[key].get_path())
        for key, test_corpus in test_corpora.items():
            test_corpus.dump(self.out_test_corpora[key].get_path())
