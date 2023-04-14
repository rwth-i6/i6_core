import h5py
import numpy as np
import logging, sys, shutil, tempfile

from typing import Dict, List, Optional


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


class NextGenHDFWriter:
    """
    This class is a helper for writing the of returnn NextGenHDFDataset
    """

    def __init__(
        self,
        filename: str,
        label_info_dict: Dict,
        feature_names: Optional[List[str]] = None,
        label_data_type: type = np.uint16,
        label_parser_name: str = "sparse",
        feature_parser_name: str = "feature_sequence",
    ):
        """
        :param label_info_dict: a dictionay with the label targets used in returnn training as key and numebr of label classes as value
        :param feature_names: additional feature data names
        :param label_data_type: type that is used to store the data
        :param label_parser_name: this should be checked against returnn implementations
        "param feature_parser_name: as above
        """
        self.label_info_dict = label_info_dict
        self.label_parser_name = label_parser_name
        self.feature_names = feature_names
        if feature_names is not None:
            self.feature_parser_name = feature_parser_name
        self.label_data_type = label_data_type
        self.string_data_type = h5py.special_dtype(vlen=str)
        self.sequence_names = []
        self.group_holder_dict = {}

        self.file_init()

    def file_init(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix="_NextGenHDFWriter_outHDF")
        self.temp_path = self.temp_file.name
        self.out_hdf = h5py.File(self.temp_path, "w")

        logging.info(f"processing temporary file { self.temp_path}")

        # root
        self.root_group = self.out_hdf.create_group("streams")

        for label_name, label_dim in self.label_info_dict.items():
            self.group_holder_dict[label_name] = self._get_label_group(label_name, label_dim)

        if self.feature_names is not None:
            for feat_name in self.feature_names:
                self.group_holder_dict[feat_name] = self._get_feature_group(feat_name)

    def _get_label_group(self, label_name, label_dim):
        assert label_dim > 0, "you should have at least dim 1"
        label_group = self.root_group.create_group(label_name)
        label_group.attrs["parser"] = "sparse"
        label_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(label_dim)],
            dtype=self.string_data_type,
        )

        return label_group.create_group("data")

    def _get_feature_group(self, feature_name):
        feature_group = self.root_group.create_group(feature_name)
        feature_group.attrs["parser"] = self.feature_parser_name

        return feature_group.create_group("data")

    def add_sequence_name(self, seq_name):
        self.sequence_names.append(seq_name)

    def add_data_to_group(self, group_name, seq_name, data):
        if group_name in self.label_info_dict:
            data = np.array(data).astype(self.label_data_type)

        # the / in the string would lead to more hierarchies automatically, thus substitute
        self.group_holder_dict[group_name].create_dataset(seq_name.replace("/", "\\"), data=data)

    def finalize(self, filename):
        seq_name_set = set([s.replace("/", "\\") for s in self.sequence_names])

        for k, group in self.group_holder_dict.items():
            assert set(group.keys()) == seq_name_set, "The sequence names do not match between groups"

        self.out_hdf.create_dataset(
            "seq_names", data=[s.encode() for s in self.sequence_names], dtype=self.string_data_type
        )

        self.out_hdf.close()
        shutil.move(self.temp_path, filename)
