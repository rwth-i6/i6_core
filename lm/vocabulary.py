__all__ = [
    "LmIndexVocabulary",
    "LmIndexVocabularyFromLexiconJob",
    "VocabularyFromLmJob",
    "VocabularyFromTextJob",
]
from sisyphus import Job, Task, tk

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union

from i6_core.lib.lm import Lm
from i6_core.lib.lexicon import Lexicon
from i6_core.util import uopen


@dataclass(frozen=True)
class LmIndexVocabulary:
    vocab: tk.Path
    vocab_size: tk.Variable
    unknown_token: Union[tk.Variable, str]


class LmIndexVocabularyFromLexiconJob(Job):
    """
    Computes a <word>: <index> vocabulary file from a bliss lexicon for Word-Level LM training

    Sentence begin/end will point to index 0, unknown to index 1.
    Both are taking directly from the lexicon via the "special" marking:
      - <lemma special="sentence-begin"> -> index 0
      - <lemma special="sentence-end"> -> index 0
      - <lemma special="unknown"> -> index 1

    If <synt> tokens are provided in a lemma, they will be used instead of <orth>

    CAUTION:
    Be aware of: https://github.com/rwth-i6/returnn/issues/1245 when using Returnn's LmDataset
    """

    def __init__(self, bliss_lexicon: tk.Path, count_ordering_text: Optional[tk.Path] = None):
        """
        :param bliss_lexicon: us the lemmas from the lexicon to define the indices
        :param count_ordering_text: optional text that can be used to define the index order based on the lemma count
        """
        self.bliss_lexicon = bliss_lexicon
        self.count_ordering_text = count_ordering_text

        self.out_vocab = self.output_path("lm.vocab.txt")
        self.out_vocab_size = self.output_var("vocab_size")
        self.out_unknown_token = self.output_var("unknown_token")

        self.out_vocabulary_object = LmIndexVocabulary(
            vocab=self.out_vocab,
            vocab_size=self.out_vocab_size,
            unknown_token=self.out_unknown_token,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        word_counts = OrderedDict()
        sentence_begin = None
        sentence_end = None
        sentence_boundary = None
        unknown = None

        for lemma in lex.lemmata:
            if lemma.special == "sentence-begin":
                sentence_begin_list = lemma.synt if lemma.synt is not None else lemma.orth
                assert len(sentence_begin_list) == 1
                sentence_begin = sentence_begin_list[0]
            elif lemma.special == "sentence-end":
                sentence_end_list = lemma.synt if lemma.synt is not None else lemma.orth
                assert len(sentence_end_list) == 1
                sentence_end = sentence_end_list[0]
            elif lemma.special == "sentence-boundary":
                sentence_boundary_list = lemma.synt if lemma.synt is not None else lemma.orth
                assert len(sentence_boundary_list) == 1
                sentence_boundary = sentence_boundary_list[0]
            elif lemma.special == "unknown":
                unknown_list = lemma.synt if lemma.synt is not None else lemma.orth
                assert len(unknown_list) == 1
                unknown = unknown_list[0]
            elif lemma.synt is not None:
                for synt in lemma.synt:
                    word_counts[synt] = 0
            else:
                for orth in lemma.orth:
                    word_counts[orth] = 0

        assert sentence_boundary is not None or (sentence_begin is not None and sentence_end is not None)
        assert unknown is not None

        if self.count_ordering_text is not None:
            with uopen(self.count_ordering_text, "rt") as f:
                for line in f.readlines():
                    for word in line.strip().split(" "):
                        if word in word_counts:
                            word_counts[word] += 1
            wordlist = [w for w, _ in sorted(word_counts.items(), key=lambda wc: wc[1], reverse=True)]
        else:
            wordlist = [word for word in word_counts.keys()]

        with uopen(self.out_vocab, "wt") as f:
            if sentence_begin is not None:
                f.write("%s 0\n" % sentence_begin)
            if sentence_end is not None:
                f.write("%s 0\n" % sentence_end)
            if sentence_boundary is not None:
                f.write("%s 0\n" % sentence_boundary)
            f.write("%s 1\n" % unknown)
            for i, word in enumerate(wordlist):
                f.write("%s %d\n" % (word, i + 2))
            self.out_vocab_size.set(i + 3)

        self.out_unknown_token.set(unknown)


class VocabularyFromLmJob(Job):
    """
    Extract the vocabulary from an existing LM. Currently supports only arpa files for input.
    """

    def __init__(self, lm_file):
        """
        :param Path lm_file: path to the lm arpa file
        """
        self.lm_path = lm_file
        self.out_vocabulary = self.output_path("vocabulary.txt")
        self.out_vocabulary_size = self.output_var("vocabulary_size")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        lm = Lm(self.lm_path.get())
        self.out_vocabulary_size.set(lm.ngram_counts[0])

        vocabulary = set()

        for n in range(len(lm.ngram_counts)):
            for words, _ in lm.get_ngrams(n + 1):
                for word in words.split(" "):
                    vocabulary.add(word)

        with open(self.out_vocabulary.get_path(), "w") as fout:
            for word in sorted(vocabulary):
                fout.write(f"{word}\n")


class VocabularyFromTextJob(Job):
    """
    Extract vocabulary from given text files based on frequency.
    """

    def __init__(self, file_paths: List[tk.Path], num_words: int = 1_000_000, include_counts: bool = False):
        """
        :param file_paths: paths to the text files
        :param num_words: expected size of the vocabulary
        :param include_counts: whether write the counts of the words into the vocabulary file
        """
        self.file_paths = file_paths
        self.num_words = num_words
        self.include_counts = include_counts

        self.out_vocabulary = self.output_path("vocabulary.txt")
        self.out_counter = self.output_var("counter")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        counter = Counter()

        for file_path in self.file_paths:
            with uopen(file_path, "rt") as input_file:
                for line in input_file:
                    words = line.strip().split()
                    counter.update(words)

        cutoff = min(self.num_words, len(counter))

        with open(self.out_vocabulary, "w") as vocabulary:
            if self.include_counts:
                for (word, count) in counter.most_common(cutoff):
                    vocabulary.write(f"{word} {count}\n")
            else:
                for (word, _) in counter.most_common(cutoff):
                    vocabulary.write(f"{word}\n")

        return counter
