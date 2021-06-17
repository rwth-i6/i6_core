__all__ = ["BlissToOggZipJob"]

import os
import subprocess as sp

from i6_core.util import relink

from sisyphus import *

Path = setup_path(__package__)


class BlissToOggZipJob(Job):
    """
    This Job is a wrapper around the RETURNN tool bliss-to-ogg-zip.py.

    """

    def __init__(
        self,
        bliss_corpus,
        segments=None,
        rasr_cache=None,
        raw_sample_rate=None,
        feat_sample_rate=None,
        no_conversion=False,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """
        use RETURNN to dump data into an ogg zip file

        :param str|Path bliss_corpus: bliss corpus file
        :param str|Path segments: RASR segment file
        :param str|Path rasr_cache: feature rasr cache
        :param int raw_sample_rate: raw audio sampling rate
        :param int feat_sample_rate: feature sampling rate
        :param bool no_conversion: do not call the actual conversion, assume the audio files are already correct
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """
        self.bliss_corpus = bliss_corpus
        self.segments = segments
        self.rasr_cache = rasr_cache
        self.raw_sample_rate = raw_sample_rate
        self.feat_sample_rate = feat_sample_rate
        self.no_conversion = no_conversion

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_ogg_zip = self.output_path("out.ogg.zip")

        self.rqmt = None

    def tasks(self):
        if self.rqmt:
            yield Task("run", rqmt=self.rqmt)
        else:
            yield Task("run", mini_task=True)

    def run(self):
        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(
                tk.uncached_path(self.returnn_root), "tools/bliss-to-ogg-zip.py"
            ),
            tk.uncached_path(self.bliss_corpus),
            "--output",
            "out.ogg.zip",
        ]
        if self.segments is not None:
            args.extend(["--subset_segment_file", tk.uncached_path(self.segments)])

        if self.no_conversion:
            args.extend(["--no_conversion"])
        else:
            if self.rasr_cache is not None:
                args.extend(["--sprint_cache", tk.uncached_path(self.rasr_cache)])
            if self.raw_sample_rate is not None:
                args.extend(["--raw_sample_rate", str(self.raw_sample_rate)])
            if self.feat_sample_rate is not None:
                args.extend(["--feat_sample_rate", str(self.feat_sample_rate)])

        sp.check_call(args)
        relink("out.ogg.zip", self.out_ogg_zip.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["returnn_python_exe"]
        return super().hash(parsed_args)
