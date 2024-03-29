import h5py
import numpy as np
from typing import Dict, Optional
import sys


def get_input_dict_from_returnn_hdf(hdf_file: h5py.File) -> Dict[str, np.ndarray]:
    """
    Generate dictionary containing the "data" value as ndarray indexed by the sequence tag

    :param hdf_file: HDF file to extract data from
    :return:
    """
    inputs = hdf_file["inputs"]
    seq_tags = hdf_file["seqTags"]
    lengths = hdf_file["seqLengths"]

    output_dict = {}
    offset = 0
    for tag, length in zip(seq_tags, lengths):
        tag = tag if isinstance(tag, str) else tag.decode()
        output_dict[tag] = inputs[offset : offset + length[0]]
        offset += length[0]

    return output_dict


def get_returnn_simple_hdf_writer(returnn_root: Optional[str]):
    """
    Get the RETURNN SimpleHDFWriter, will add return to the path, so only use in Job runtime
    :param returnn_root:
    """
    if returnn_root:
        sys.path.append(returnn_root)
    from returnn.datasets.hdf import SimpleHDFWriter

    return SimpleHDFWriter
