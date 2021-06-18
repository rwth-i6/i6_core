__all__ = ["PipelineJob", "ConcatenateJob", "HeadJob", "TailJob"]

import os
from sisyphus import Job, Task, Path, global_settings as gs


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
        :param list[str] pipeline: list of shell commands to form the pipeline
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
            inputs = " ".join(gs.file_caching(str(i)) for i in self.text_files)
        else:
            inputs = gs.file_caching(str(self.text_files))
        self.sh("zcat -f %s | %s > %s" % (inputs, pipe, self.out.get_path()))

        # assume that we do not want empty pipe results
        assert not (os.stat(str(self.out)).st_size == 0), "Pipe result was empty"

        input_length = int(self.sh("zcat -f %s | sed '$a\\' | wc -l" % inputs, True))
        assert input_length > 0
        output_length = int(self.sh("zcat -f %s | wc -l" % self.out.get_path(), True))
        assert output_length > 0
        if self.check_equal_length:
            assert (
                input_length == output_length
            ), "pipe input and output lengths do not match"

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

    def __init__(self, text_files):
        """
        :param list[Path] text_files: input text files
        """
        assert text_files

        # ensure sets are always merged in the same order
        if isinstance(text_files, set):
            text_files = list(text_files)
            text_files.sort(key=lambda x: str(x))

        assert isinstance(text_files, list)

        # Skip this job if only one input is present
        if len(text_files) == 1:
            self.out = text_files.pop()
        else:
            self.out = self.output_path("out.gz")

        for input in text_files:
            assert isinstance(input, Path) or isinstance(
                input, str
            ), "input to Concatenate is not a valid path"

        self.text_files = text_files

    def tasks(self):
        yield Task("run", rqmt={"mem": 3, "time": 3})

    def run(self):
        self.f_list = " ".join(gs.file_caching(str(i)) for i in self.text_files)
        self.sh("zcat -f {f_list} | gzip > {out}")


class HeadJob(Job):
    """
    Return the head of a text file, either absolute or as ratio (provide one)
    """

    def __init__(self, text_file, num_lines=None, ratio=None):
        """
        :param Path text_file: text file (gz or raw)
        :param int num_lines: number of lines to extract
        :param float ratio: ratio of lines to extract
        """

        assert num_lines or ratio, "please specify either lines or ratio"
        assert not (num_lines and ratio), "please specify only lines or ratio, not both"
        self.set_attrs(locals())
        if ratio:
            assert ratio <= 1
        self.out = self.output_path("out.gz")
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
            assert not self.lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.lines = int(length * self.ratio)

        self.sh(
            "zcat -f {data} | head -n {num_lines} | gzip > {out}",
            except_return_codes=(141,),
        )
        self.length.set(self.lines)


class TailJob(HeadJob):
    """
    Return the tail of a text file, either absolute or as ratio (provide one)
    """

    def run(self):
        if self.ratio:
            assert not self.lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.lines = int(length * self.ratio)

        self.sh("zcat -f {data} | tail -n {lines} | gzip > {out}")
