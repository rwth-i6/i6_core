__all__ = ["Pipeline", "Concatenate", "Head", "Tail"]

import os
from sisyphus import Job, Task, Path, global_settings as gs


class Pipeline(Job):
    """
    Reads a text file and applies a list of piped shell commands

    :param Path text: text file (raw or gz) to be processed
    :param list[str] pipeline: list of shell commands to form the pipeline
    :param bool zip_out: apply gzip to the output
    :param bool check_equal_length: the line count of the input and output should match
    :param bool mini_task: the pipeline should be run as mini_task
    """

    def __init__(
        self, text, pipeline, zip_out=False, check_equal_length=False, mini_task=False
    ):
        assert text is not None
        self.set_attrs(locals())
        if zip_out:
            self.out = self.output_path("out.gz")
        else:
            self.out = self.output_path("out")

        self.check_equal_length = check_equal_length
        self.pipeline = pipeline

    def tasks(self):
        if isinstance(self.text, (list, tuple)):
            size = sum(text.estimate_text_size() / 1024 for text in self.text)
        else:
            size = self.text.estimate_text_size() / 1024

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
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt={"time": time, "mem": mem, "cpu": cpu})

    def run(self):
        pipeline = self.pipeline.copy()
        if self.zip_out:
            pipeline.append("gzip")
        self.pipe = " | ".join([str(i) for i in pipeline])
        if isinstance(self.text, (list, tuple)):
            self.input_text = " ".join(gs.file_caching(str(i)) for i in self.text)
        else:
            self.input_text = gs.file_caching(str(self.text))
        self.sh("zcat -f {input_text} | {pipe} > {out}")

        # assume that we do not want empty pipe results
        assert not (os.stat(str(self.out)).st_size == 0), "Pipe result was empty"

        input_length = int(self.sh("zcat -f {input_text} | sed '$a\\' | wc -l", True))
        assert input_length > 0
        output_length = int(self.sh("zcat -f {out} | wc -l", True))
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
        return super(Pipeline, cls).hash(args)


class Concatenate(Job):
    """
    Concatenate all given input files (gz or raw)
    :param list[Path] inputs: input text files
    """

    def __init__(self, inputs):
        assert inputs

        # ensure sets are always merged in the same order
        if isinstance(inputs, set):
            inputs = list(inputs)
            inputs.sort(key=lambda x: str(x))

        assert isinstance(inputs, list)

        # Skip this job if only one input is present
        if len(inputs) == 1:
            self.out = inputs.pop()
        else:
            self.out = self.output_path("out.gz")

        for input in inputs:
            assert isinstance(input, Path) or isinstance(
                input, str
            ), "input to Concatenate is not a valid path"

        self.inputs = inputs

    def tasks(self):
        yield Task("run", rqmt={"mem": 3, "time": 3})

    def run(self):
        self.f_list = " ".join(gs.file_caching(str(i)) for i in self.inputs)
        self.sh("zcat -f {f_list} | gzip > {out}")


class Head(Job):
    """
    Return the head of a text file, either absolute or as ratio (provide one)
    :param Path data: text file (gz or raw)
    :param int lines: number of lines to extract
    :param float ratio: ratio of lines to extract
    """

    def __init__(self, data, lines=None, ratio=None):

        assert lines or ratio, "please specify either lines or ratio"
        assert not (lines and ratio), "please specify only lines or ratio, not both"
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
            length = int(self.sh("zcat -f {data} | wc -l", True))
            self.lines = int(length * self.ratio)

        self.sh(
            "zcat -f {data} | head -n {lines} | gzip > {out}",
            except_return_codes=(141,),
        )
        self.length.set(self.lines)


class Tail(Head):
    def run(self):
        if self.ratio:
            assert not self.lines
            length = int(self.sh("zcat -f {data} | wc -l", True))
            self.lines = int(length * self.ratio)

        self.sh("zcat -f {data} | tail -n {lines} | gzip > {out}")
