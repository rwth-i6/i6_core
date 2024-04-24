__all__ = [
    "TextDictToTextLinesJob",
]

from sisyphus import Job, Path, Task
from i6_core.util import uopen


class TextDictToTextLinesJob(Job):
    """
    Operates on RETURNN search output (see :mod:`i6_core.returnn.search`)
    or :class:`CorpusToTextDictJob` output and prints the values line-by-line.
    The ordering from the dict is preserved.
    """

    def __init__(self, text_dict: Path, *, gzip: bool = False):
        """
        :param text_dict: a text file with a dict in python format, {seq_tag: text}
        :param gzip: if True, gzip the output
        """
        self.text_dict = text_dict
        self.out_text_lines = self.output_path("text_lines.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        d = eval(uopen(self.text_dict, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> text

        with uopen(self.out_text_lines, "wt") as out:
            for seq_tag, entry in d.items():
                assert isinstance(entry, str), f"expected str, got {entry!r} (type {type(entry).__name__})"
                out.write(entry + "\n")
