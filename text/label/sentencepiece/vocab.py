"""
Jobs related to the SPM vocabulary.
"""

from sisyphus import Job, Task, tk


class ExtractSentencePieceVocabJob(Job):
    """
    Extract the vocab from a sentence piece model (SPM)
    """

    def __init__(self, model: tk.Path):
        super().__init__()
        self.model = model
        self.out_vocab = self.output_path("spm.vocab")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sentencepiece

        sp = sentencepiece.SentencePieceProcessor(model_file=self.model.get_path())
        with open(self.out_vocab.get_path(), "w") as f:
            f.write("{\n")
            for i in range(sp.vocab_size()):
                f.write(f"{sp.id_to_piece(i)!r}: {i},\n")
            f.write("}\n")
