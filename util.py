from collections.abc import Mapping
import gzip
import logging
import os
import shutil
import stat
import subprocess as sp
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Union

from sisyphus import *
from sisyphus.delayed_ops import DelayedBase, DelayedFormat

Path = setup_path(__package__)
Variable = tk.Variable


class MultiPath:
    def __init__(
        self,
        path_template,
        hidden_paths,
        cached=False,
        path_root=None,
        hash_overwrite=None,
    ):
        self.path_template = path_template
        self.hidden_paths = hidden_paths
        self.cached = cached
        self.path_root = path_root
        self.hash_overwrite = hash_overwrite

    def __str__(self):
        if self.path_root is not None:
            result = os.path.join(self.path_root, self.path_template)
        else:
            result = self.path_template
        if self.cached:
            result = gs.file_caching(result)
        return result

    def __sis_state__(self):
        return {
            "path_template": self.path_template if self.hash_overwrite is None else self.hash_overwrite,
            "hidden_paths": self.hidden_paths,
            "cached": self.cached,
        }


class MultiOutputPath(MultiPath):
    def __init__(self, creator, path_template, hidden_paths, cached=False):
        super().__init__(
            os.path.join(creator._sis_path(gs.JOB_OUTPUT), path_template),
            hidden_paths,
            cached,
            gs.BASE_DIR,
        )


def write_paths_to_file(file: Union[str, tk.Path], paths: List[Union[str, tk.Path]]):
    with open(tk.uncached_path(file), "w") as f:
        for p in paths:
            f.write(tk.uncached_path(p) + "\n")


def zmove(src: Union[str, tk.Path], target: Union[str, tk.Path]):
    src = tk.uncached_path(src)
    target = tk.uncached_path(target)

    if not src.endswith(".gz"):
        tmp_path = src + ".gz"
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        sp.check_call(["gzip", src])
        src += ".gz"
    if not target.endswith(".gz"):
        target += ".gz"

    shutil.move(src, target)


def delete_if_exists(file: str):
    if os.path.exists(file):
        os.remove(file)


def delete_if_zero(file: str):
    if os.path.exists(file) and os.stat(file).st_size == 0:
        os.remove(file)


def backup_if_exists(file: str):
    if os.path.exists(file):
        dir, base = os.path.split(file)
        base = add_suffix(base, ".gz")
        idx = 1
        while os.path.exists(os.path.join(dir, "backup.%.4d.%s" % (idx, base))):
            idx += 1
        zmove(file, os.path.join(dir, "backup.%.4d.%s" % (idx, base)))


def remove_suffix(string: str, suffix: str) -> str:
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


def add_suffix(string: str, suffix: str) -> str:
    if not string.endswith(suffix):
        return string + suffix
    return string


def partition_into_tree(l: List, m: int) -> List[List]:
    """Transforms the list l into a nested list where each sub-list has at most length m + 1"""
    nextPartition = partition = l
    while len(nextPartition) > 1:
        partition = nextPartition
        nextPartition = []
        d = len(partition) // m
        mod = len(partition) % m
        if mod <= d:
            p = 0
            for i in range(mod):
                nextPartition.append(partition[p : p + m + 1])
                p += m + 1
            for i in range(d - mod):
                nextPartition.append(partition[p : p + m])
                p += m
            assert p == len(partition)
        else:
            p = 0
            for i in range(d):
                nextPartition.append(partition[p : p + m])
                p += m
            nextPartition.append(partition[p : p + mod])
            assert p + mod == len(partition)
    return partition


def reduce_tree(func, tree):
    return func([(reduce_tree(func, e) if type(e) == list else e) for e in tree])


def uopen(path: Union[str, tk.Path], *args, **kwargs) -> Union[gzip.open, open]:
    path = tk.uncached_path(path)
    if path.endswith(".gz"):
        return gzip.open(path, *args, **kwargs)
    else:
        return open(path, *args, **kwargs)


def get_val(var: Any) -> Any:
    if isinstance(var, Variable):
        return var.get()
    return var


def num_cart_labels(path: Union[str, tk.Path]) -> int:
    path = tk.uncached_path(path)
    if path.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    file = open_func(path, "rt")
    tree = ET.parse(file)
    file.close()
    all_nodes = tree.findall("binary-tree//node")
    return len([n for n in all_nodes if n.find("node") is None])


def chunks(l: List, n: int) -> List[List]:
    """
    :param l: list which should be split into chunks
    :param n: number of chunks
    :return: yields n chunks
    """
    bigger_count = len(l) % n
    start = 0
    block_size = len(l) // n
    for i in range(n):
        end = start + block_size + (1 if i < bigger_count else 0)
        yield l[start:end]
        start = end


def relink(src: str, dst: str):
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.link(src, dst)


def cached_path(path: Union[str, tk.Path]) -> Union[str, bytes]:
    if tk.is_path(path) and path.cached:
        caching_command = gs.file_caching(tk.uncached_path(path))
        caching_command = caching_command.replace("`", "")
        caching_command = caching_command.split(" ")
        if len(caching_command) > 1:
            ret = sp.check_output(caching_command)
            return ret.strip()
    return tk.uncached_path(path)


def write_xml(filename: Union[Path, str], element_tree: Union[ET.ElementTree, ET.Element], prettify: bool = True):
    """
    writes element tree to xml file
    :param filename: name of desired output file
    :param element_tree: element tree which should be written to file
    :param prettify: prettify the xml. Warning: be careful with this option if you care about whitespace in the xml.
    """

    def remove_unwanted_whitespace(elem):
        import re

        has_non_whitespace = re.compile(r"\S")
        for element in elem.iter():
            if not re.search(has_non_whitespace, str(element.tail)):
                element.tail = ""
            if not re.search(has_non_whitespace, str(element.text)):
                element.text = ""

    if isinstance(element_tree, ET.ElementTree):
        root = element_tree.getroot()
    elif isinstance(element_tree, ET.Element):
        root = element_tree
    else:
        assert False, "please provide an ElementTree or Element"

    if prettify:
        remove_unwanted_whitespace(root)
        xml_string = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 2)
    else:
        xml_string = ET.tostring(root, encoding="unicode")

    with uopen(filename, "wt") as f:
        f.write(xml_string)


def create_executable(filename: str, command: List[str]):
    """
    create an executable .sh file calling a single command
    :param filename: executable name ending with .sh
    :param command: list representing the command and parameters
    :return:
    """
    assert filename.endswith(".sh")
    with open(filename, "wt") as f:
        f.write("#!/usr/bin/env bash\n%s" % " ".join(command))
    os.chmod(
        filename,
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )


def compute_file_sha256_checksum(filename: str) -> str:
    """
    Computes the sha256sum for a file

    :param filename: a single file to be checked
    :return: checksum
    :rtype:str
    """
    checksum_command_output = sp.check_output(["sha256sum", filename])
    return checksum_command_output.decode().strip().split(" ")[0]


def check_file_sha256_checksum(filename: str, reference_checksum: str):
    """
    Validates the sha256sum for a file against the target checksum

    :param filename: a single file to be checked
    :param reference_checksum: checksum to verify against
    """
    assert compute_file_sha256_checksum(filename) == reference_checksum


def instanciate_delayed(o: Any) -> Any:
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    :param o: nested structure that may contain DelayedBase objects
    :return:
    """
    if isinstance(o, DelayedBase):
        o = o.get()
    elif isinstance(o, list):
        for k in range(len(o)):
            o[k] = instanciate_delayed(o[k])
    elif isinstance(o, tuple):
        o = tuple(instanciate_delayed(e) for e in o)
    elif isinstance(o, dict):
        for k in o:
            o[k] = instanciate_delayed(o[k])
    return o


already_printed_gs_warnings = set()


def get_executable_path(
    path: Optional[tk.Path],
    gs_member_name: Optional[str],
    default_exec_path: Optional[tk.Path] = None,
) -> tk.Path:
    """
    Helper function that allows to select a specific version of software while
    maintaining compatibility to different methods that were used in the past to select
    software versions.
    It will return a Path object for the first path found in

    :param path: Directly specify the path to be used
    :param gs_member_name: get path from sisyphus.global_settings.<gs_member_name>
    :param default_exec_path: general fallback if no specific version is given
    """
    global already_printed_gs_warnings
    if path is not None:
        if isinstance(path, tk.Path):
            return path
        elif isinstance(path, str):
            logging.warning(f"use of str is deprecated, please provide a Path object for {path}")
            return tk.Path(path)
        elif isinstance(path, DelayedFormat):
            logging.warning(
                f"use of a DelayedFormat is deprecated, please use Path.join_right to provide a Path object for {path}"
            )
            if (
                len(path.args) == 2
                and isinstance(path.args[0], tk.Path)
                and isinstance(path.args[1], str)
                and path.string == "{}/{}"
            ):
                return path.args[0].join_right(path.args[1])
            else:
                return tk.Path(path.get())
        assert False, f"unsupported type of {type(path)} for input {path}"
    if getattr(gs, gs_member_name, None) is not None:
        if gs_member_name not in already_printed_gs_warnings:
            logging.warning(f"use of gs is deprecated, please provide a Path object for gs.{gs_member_name}")
            already_printed_gs_warnings.add(gs_member_name)

        return tk.Path(getattr(gs, gs_member_name))
    if default_exec_path is not None:
        return default_exec_path
    assert False, f"could not find executable for {gs_member_name}"


def get_returnn_root(returnn_root: tk.Path) -> tk.Path:
    """gets the path to the root folder of RETURNN"""
    return get_executable_path(returnn_root, "RETURNN_ROOT")


def get_returnn_python_exe(returnn_python_exe: tk.Path) -> tk.Path:
    """gets the path to a python binary or script that is used to run RETURNN"""
    system_python = tk.Path(shutil.which(gs.SIS_COMMAND[0]))
    return get_executable_path(returnn_python_exe, "RETURNN_PYTHON_EXE", system_python)


def get_g2p_path(g2p_path: tk.Path) -> tk.Path:
    """gets the path to the sequitur g2p script"""
    system_python_path = os.path.dirname(shutil.which(gs.SIS_COMMAND[0]))
    system_g2p = tk.Path(system_python_path).join_right("g2p.py")
    return get_executable_path(g2p_path, "G2P_PATH", system_g2p)


def get_g2p_python(g2p_python: tk.Path) -> tk.Path:
    """gets the path to a python binary or script that is used to run g2p"""
    system_python = tk.Path(shutil.which(gs.SIS_COMMAND[0]))
    return get_executable_path(g2p_python, "G2P_PYTHON", system_python)


def get_subword_nmt_repo(subword_nmt_repo: tk.Path) -> tk.Path:
    """gets the path to the root folder of subword-nmt repo"""
    return get_executable_path(subword_nmt_repo, "SUBWORD_NMT_PATH")


def update_nested_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]):
    """updates dict 1 with all the items from dict2, both dict1 and dict2 can be nested dict"""
    for k, v in dict2.items():
        if isinstance(v, Mapping):
            dict1[k] = update_nested_dict(dict1.get(k, {}), v)
        else:
            dict1[k] = v
    return dict1


def parse_text_dict(path: Union[str, tk.Path]) -> Dict[str, str]:
    """
    Loads the text dict at :param:`path`.

    Works around https://github.com/rwth-i6/i6_core/issues/539 (``OverflowError: line number table is too long``)
    by stripping the newlines from the text dict before the ``eval``.
    """

    with uopen(path, "rt") as text_dict_file:
        txt = "".join(line.strip() for line in text_dict_file)
    d = eval(txt, {"nan": float("nan"), "inf": float("inf")})
    assert isinstance(d, dict), f"expected a text dict, but found {type(d)}"
    return d
