import logging
import shutil
import subprocess as sp
import tempfile
import os

from sisyphus import Job, Task, tk

import i6_core.util as util

try:
    import sentencepiece
except ImportError:
    if not hasattr(gs, "WARNING_NO_SENTENCEPIECE") or gs.WARNING_NO_SENTENCEPIECE is True:
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make sure it is installed "
            "in the python environment running the Sisyphus worker. To suppress this warning set "
            "'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class ApplySentencepieceToTextJob(Job):
    """
    Apply sentencepiece model on a text file, basically a wrapper for spm.encode
    """

    def __init__(
        self,
        *,
        text_file: tk.Path,
        sentencepiece_model: tk.Path,
        map_unk: bool = False,
        gzip_output: bool = True,
        mini_task: bool = True,
    ):
        """
        :param text_file: words text file to convert to sentencepiece
        :param sentencepiece_model: path to the trained sentencepiece model
        :param map_unk: when encoding string to string, spm won't map oov symbol to <unk> but keep it as is.
            This option forces the oov labels to be <unk> by cecking encoded indices.
        :param gzip_output: use gzip on the output text
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.sentencepiece_model = sentencepiece_model
        self.map_unk = map_unk

        self.out_sentencepiece_text = self.output_path(
            "words_to_sentencepiece.txt.gz" if gzip_output else "words_to_sentencepiece.txt"
        )

        self.mini_task = mini_task
        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sentencepiece

        spm = sentencepiece.SentencePieceProcessor(model_file=self.sentencepiece_model.get_path())
        unk_id = spm.unk_id()
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            input_file = self.text_file.get_path()
            tmp_infile = os.path.join(tmp, "in_text.txt")
            tmp_outfile = os.path.join(tmp, "out_text.txt")
            # normalize text format
            with util.uopen(input_file, "rt") as in_file, open(tmp_infile, "wt") as out:
                for line in in_file:
                    out.write(line)

            with util.uopen(tmp_infile, "rt") as fin, util.uopen(tmp_outfile, "wt") as fout:
                for line in fin:
                    pieces = spm.encode(line.rstrip("\n"), out_type=str)
                    if self.map_unk:
                        pieces_id = spm.encode(line.rstrip("\n"))
                        assert len(pieces_id) == len(pieces)
                        if unk_id in pieces_id:
                            pieces[:] = ["<unk>" if x == unk_id else y for x, y in zip(pieces_id, pieces)]
                    fout.write(" ".join(pieces) + "\n")

            with util.uopen(tmp_outfile, "rt") as fin, util.uopen(self.out_sentencepiece_text, "wt") as fout:
                shutil.copyfileobj(fin, fout)

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)
