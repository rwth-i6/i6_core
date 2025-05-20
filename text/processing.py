__all__ = [
    "PipelineJob",
    "ConcatenateJob",
    "HeadJob",
    "TailJob",
    "SetDifferenceJob",
    "WriteToTextFileJob",
    "WriteToCsvFileJob",
    "SplitTextFileJob",
]

import csv
from io import IOBase
import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
import tempfile
from typing import Dict, List, Optional, Union

from sisyphus import Job, Task, Path, global_settings as gs, toolkit as tk
from sisyphus.delayed_ops import DelayedBase

import i6_core.util as util


class PipelineJob(Job):
    """
    Reads a text file and applies a list of piped shell commands
    """

    def __init__(
        self,
        text_files,
        pipeline,
        zip_output=False,
        check_equal_length=False,
        mini_task=False,
    ):
        """
        :param iterable[Path]|Path text_files: text file (raw or gz) or list of files to be processed
        :param list[str|DelayedBase] pipeline: list of shell commands to form the pipeline,
            can be empty to use the job for concatenation or gzip compression only.
        :param bool zip_output: apply gzip to the output
        :param bool check_equal_length: the line count of the input and output should match
        :param bool mini_task: the pipeline should be run as mini_task
        """
        assert text_files is not None
        self.text_files = text_files
        self.pipeline = pipeline
        self.zip_output = zip_output
        self.check_equal_length = check_equal_length
        self.mini_task = mini_task

        if zip_output:
            self.out = self.output_path("out.gz")
        else:
            self.out = self.output_path("out")

        self.rqmt = None

    def tasks(self):
        if not self.rqmt:
            # estimate rqmt if not set explicitly
            if isinstance(self.text_files, (list, tuple)):
                size = sum(text.estimate_text_size() / 1024 for text in self.text_files)
            else:
                size = self.text_files.estimate_text_size() / 1024

            if size <= 128:
                time = 2
                mem = 2
            elif size <= 512:
                time = 3
                mem = 3
            elif size <= 1024:
                time = 4
                mem = 3
            elif size <= 2048:
                time = 6
                mem = 4
            else:
                time = 8
                mem = 4
            cpu = 1
            self.rqmt = {"time": time, "mem": mem, "cpu": cpu}

        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        pipeline = self.pipeline.copy()
        if self.zip_output:
            pipeline.append("gzip")
        pipe = " | ".join([str(i) for i in pipeline])
        if isinstance(self.text_files, (list, tuple)):
            inputs = " ".join(i.get_cached_path() for i in self.text_files)
        else:
            inputs = self.text_files.get_cached_path()
        if pipe:
            self.sh("zcat -f %s | %s > %s" % (inputs, pipe, self.out.get_path()))
        else:
            self.sh("zcat -f %s > %s" % (inputs, self.out.get_path()))

        # assume that we do not want empty pipe results
        assert not (os.stat(str(self.out)).st_size == 0), "Pipe result was empty"

        input_length = int(self.sh("zcat -f %s | sed '$a\\' | wc -l" % inputs, True))
        assert input_length > 0
        output_length = int(self.sh("zcat -f %s | wc -l" % self.out.get_path(), True))
        assert output_length > 0
        if self.check_equal_length:
            assert input_length == output_length, "pipe input and output lengths do not match"

    @classmethod
    def hash(cls, parsed_args):
        args = parsed_args.copy()
        del args["check_equal_length"]
        del args["mini_task"]
        return super(PipelineJob, cls).hash(args)


class ConcatenateJob(Job):
    """
    Concatenate all given input files (gz or raw)
    """

    def __init__(self, text_files: List[Path], zip_out: bool = True, out_name: str = "out"):
        """
        :param text_files: input text files
        :param zip_out: apply gzip to the output
        :param out_name: user specific file name for the output file
        """
        assert text_files

        # ensure sets are always merged in the same order
        if isinstance(text_files, set):
            text_files = list(text_files)
            text_files.sort(key=lambda x: str(x))

        assert isinstance(text_files, list)

        for input_file in text_files:
            assert isinstance(input_file, (Path, str)), "input to Concatenate is not a valid path"

        self.text_files = text_files
        self.zip_out = zip_out

        # Skip this job if only one input is present
        if len(text_files) == 1:
            self.out = text_files.pop()
        else:
            if zip_out:
                self.out = self.output_path(out_name + ".gz")
            else:
                self.out = self.output_path(out_name)

    def tasks(self):
        yield Task("run", rqmt={"mem": 3, "time": 3})

    def run(self):
        f_list = [
            gs.file_caching(text_file) if isinstance(text_file, str) else text_file.get_cached_path()
            for text_file in self.text_files
        ]

        with util.uopen(self.out, "wb") as out_file:
            for f in f_list:
                logging.info(f)
                with util.uopen(f, "rb") as in_file:
                    shutil.copyfileobj(in_file, out_file)


class HeadJob(Job):
    """
    Return the head of a text file, either absolute or as ratio (provide one)
    """

    __sis_hash_exclude__ = {"zip_output": True}

    def __init__(self, text_file, num_lines=None, ratio=None, zip_output=True):
        """
        :param Path text_file: text file (gz or raw)
        :param int num_lines: number of lines to extract
        :param float ratio: ratio of lines to extract
        """
        assert num_lines or ratio, "please specify either lines or ratio"
        assert not (num_lines and ratio), "please specify only lines or ratio, not both"
        if ratio:
            assert ratio <= 1

        self.text_file = text_file
        self.num_lines = num_lines
        self.ratio = ratio
        self.zip_output = zip_output

        self.out = self.output_path("out.gz") if self.zip_output else self.output_path("out")
        if not self.zip_output:
            self.out.hash_overwrite = (self, "out.gz")  # keep old hashing behavior
        self.length = self.output_var("length")

    def tasks(self):
        yield Task(
            "run",
            rqmt={
                "cpu": 1,
                "time": 2,
                "mem": 4,
            },
        )

    def run(self):
        if self.ratio:
            assert not self.num_lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.num_lines = int(length * self.ratio)

        pipeline = "zcat -f {text_file} | head -n {num_lines}"
        if self.zip_output:
            pipeline += " | gzip"
        pipeline += " > {out}"

        self.sh(
            pipeline,
            except_return_codes=(141,),
        )
        self.length.set(self.num_lines)


class TailJob(HeadJob):
    """
    Return the tail of a text file, either absolute or as ratio (provide one)
    """

    def run(self):
        if self.ratio:
            assert not self.num_lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.num_lines = int(length * self.ratio)

        pipeline = "zcat -f {text_file} | tail -n {num_lines}"
        if self.zip_output:
            pipeline += " | gzip"
        pipeline += " > {out}"

        self.sh(pipeline)
        self.length.set(self.num_lines)


class SetDifferenceJob(Job):
    """
    Return the set difference of two text files, where one line is one element.
    """

    def __init__(self, minuend, subtrahend, gzipped=False):
        """
        This job performs the set difference minuend - subtrahend. Unlike the bash utility comm, the two files
        do not need to be sorted.
        :param Path minuend: left-hand side of the set subtraction
        :param Path subtrahend: right-hand side of the set subtraction
        :param bool gzipped: whether the output should be compressed in gzip format
        """
        self.minuend = minuend
        self.subtrahend = subtrahend

        outfile_ext = "txt.gz" if gzipped else "txt"
        self.out_file = self.output_path("diff.%s" % outfile_ext)

        self.rqmt = {"cpu": 1, "time": 1, "mem": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with util.uopen(self.minuend, "rt") as fin:
            file_set1 = set(fin.read().split("\n"))
        with util.uopen(self.subtrahend, "rt") as fin:
            file_set2 = set(fin.read().split("\n"))
        with util.uopen(self.out_file, "wt") as fout:
            fout.write("\n".join(sorted(file_set1.difference(file_set2))))


class WriteToTextFileJob(Job):
    """
    Write a given content into a text file, one entry per line.

    This job supports multiple input types:
    1. String.
    2. Dictionary.
    3. Iterable.

    The corresponding output for each of the inputs above is:
    1. The string is directly written into the file.
    2. Each key/value pair is written as `<key>: <value>`.
    3. Each element in the iterable is written in a separate line as a string.
    """

    __sis_hash_exclude__ = {"out_name": "file.txt"}

    def __init__(self, content: Union[str, dict, Iterable, DelayedBase], out_name: str = "file.txt"):
        """
        :param content: input which will be written into a text file
        :param out_name: user specific file name for the output file
        """
        self.content = content

        self.out_file = self.output_path(out_name)

    def write_content_to_file(self, file_handler: IOBase):
        content = util.instanciate_delayed(self.content)
        if isinstance(content, str):
            file_handler.write(content)
        elif isinstance(content, dict):
            for key, val in content.items():
                file_handler.write(f"{key}: {val}\n")
        elif isinstance(content, Iterable):
            for line in content:
                file_handler.write(f"{line}\n")
        else:
            raise NotImplementedError("Content of unknown type different from (str, dict, Iterable).")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_file.get_path(), "w") as f:
            self.write_content_to_file(f)


class WriteToCsvFileJob(WriteToTextFileJob):
    """
    Write a given content into a csv file, one entry per line.

    This job only supports dictionaries as input type. Each key/value pair is written as `<key><delimiter><value>`.
    """

    __sis_hash_exclude__ = {}  # It was filled in the base class, but it's not needed anymore since this is a new job.

    def __init__(
        self,
        content: Dict[str, Union[str, List[str]]],
        out_name: str = "file.txt",
        delimiter: str = "\t",
    ):
        """
        :param content: input which will be written into a text file
        :param out_name: user specific file name for the output file
        :param delimiter: Delimiter used to separate the different entries.
        """
        super().__init__(content, out_name)

        self.delimiter = delimiter

    def write_content_to_file(self, file_handler: IOBase):
        """
        Writes the input contents (from `self.content`) into the file provided as parameter as a csv file.

        :param file_handler: Open file to write the contents of `self.content` to.
        """
        csv_writer = csv.writer(file_handler, delimiter=self.delimiter)
        content = util.instanciate_delayed(self.content)
        if isinstance(content, dict):
            for key, val in content.items():
                if isinstance(val, list):
                    csv_writer.writerow((key, *val))
                else:
                    csv_writer.writerow((key, val))
        else:
            raise NotImplementedError("Content of unknown type different from (str, dict, Iterable).")


class SplitTextFileJob(Job):
    def __init__(
        self,
        text_file: tk.Path,
        num_lines_per_split: int,
        num_text_file_lines: Optional[int] = None,
        zip_output: bool = True,
    ):
        """
        Job splits a text file into several smaller files.

        https://stackoverflow.com/a/45761990/2062195

        :param text_file: Input text file to be processed.
        :param num_lines_per_split: Number of lines per split.
        :param num_text_file_lines: Number of lines in the input text file.
        :param zip_output: compress the output files.
        """
        self.in_text_file = text_file
        self.num_lines_per_split = num_lines_per_split
        self.num_text_file_lines = num_text_file_lines
        self.zip_output = zip_output

        if num_text_file_lines is not None:
            self.num_output_files = self.num_text_file_lines // self.num_lines_per_split + int(
                bool(self.num_text_file_lines % self.num_lines_per_split)
            )
        else:
            raise NotImplementedError

        self.out_split_text_files = {
            k: self.output_path(f"split.{k:04}.{'txt.gz' if zip_output else 'txt'}")
            for k in range(1, self.num_output_files + 1)
        }

        self.run_rqmt = {"cpu": 1, "mem": 12.0, "time": 6.0}

    def tasks(self):
        yield Task("run", rqmt=self.run_rqmt)

    def run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.in_text_file.get_path().endswith(".gz"):
                logging.info("Un-compressing file")
                text_file = f"{tmp_dir}/input_file.txt"
                with open(text_file, "wt") as f_in:
                    uncompress_cmd = ["gzip", "-cdk", self.in_text_file.get_path()]
                    subprocess.run(uncompress_cmd, check=True, stdout=f_in)
            else:
                text_file = self.in_text_file.get_path()

            logging.info("Split lines")
            split_cmd = [
                "split",
                "-l",
                str(self.num_lines_per_split),
                "--suffix-length=4",
                "--numeric-suffixes=1",
                "--additional-suffix=.txt",
                text_file,
                f"{tmp_dir}/split.",
            ]
            subprocess.run(split_cmd, check=True)

            for file_id in range(1, self.num_output_files + 1):
                file_path = f"{tmp_dir}/split.{file_id:04}.txt"
                assert os.path.isfile(file_path) and os.path.getsize(file_path) > 0

            if self.zip_output:
                logging.info("Compressing file")
                compress_cmd = ["gzip"] + [
                    f"{tmp_dir}/split.{file_id:04}.txt" for file_id in range(1, self.num_output_files + 1)
                ]
                subprocess.run(compress_cmd, check=True)

            for file_id in range(1, self.num_output_files + 1):
                shutil.move(
                    f"{tmp_dir}/split.{file_id:04}.txt.gz" if self.zip_output else f"split.{file_id:04}.txt",
                    self.out_split_text_files[file_id].get_path(),
                )
