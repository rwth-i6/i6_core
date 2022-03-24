__all__ = ["TrainBPEModelJob", "ReturnnTrainBpeJob"]

from i6_core.label.bpe.train import TrainBPEModelJob as _TrainBPEModelJob
from i6_core.label.bpe.train import ReturnnTrainBpeJob as _ReturnnTrainBpeJob


class TrainBPEModelJob(_TrainBPEModelJob):
    """
    Create a bpe codes file using the official subword-nmt repo, either installed from pip
    or https://github.com/rsennrich/subword-nmt
    """


class ReturnnTrainBpeJob(_ReturnnTrainBpeJob):
    """
    Create Bpe codes and vocab files compatible with RETURNN BytePairEncoding
    Repository:
        https://github.com/albertz/subword-nmt
    """
