"""
Tests for serialization_v2.
"""

import os
from typing import Callable
import textwrap
import functools
from dataclasses import dataclass

from i6_core.serialization_v2 import serialize_config, SisPathHandling, PyCode
from returnn.tensor import Dim, batch_dim
from sisyphus.hash import sis_hash_helper


def test_basic():
    assert serialize_config({"var1": 42, "var2": "foo"}).as_serialized_code() == "var1 = 42\nvar2 = 'foo'\n"


def test_recursive():
    d_base = {"key": 1}
    d_other = {"key": 2, "base": d_base}
    # It should serialize d_base first, even when we have d_other first here in the dict.
    assert serialize_config({"first": d_other, "second": d_base}).as_serialized_code() == textwrap.dedent(
        """\
        first_base = {'key': 1}
        first = {'key': 2, 'base': first_base}
        second = first_base
        """
    )


def test_inlining():
    d = {"d": {"k1": 1, "k2": {"k3": 3, "k4": 4}}}
    assert serialize_config(d).as_serialized_code() == f"d = {d['d']!r}\n"
    assert serialize_config(d, inlining=False).as_serialized_code() == textwrap.dedent(
        """\
        d_k2 = {'k3': 3, 'k4': 4}
        d = {'k1': 1, 'k2': d_k2}
        """
    )


def test_builtin():
    d = {"func": sum}
    assert serialize_config(d).as_serialized_code() == "func = sum\n"


def test_builtin_as_is():
    d = {"sum": sum}
    assert serialize_config(d).as_serialized_code() == "sum = sum\n"  # might change in the future...


def test_builtin_overwrite():
    d = {"sum": 42, "func": sum}
    assert serialize_config(d).as_serialized_code() == "sum = 42\nfrom builtins import sum as func\n"


def test_func():
    import i6_core
    from i6_core.util import uopen

    mod_filename = i6_core.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"test_func": uopen}
    assert serialize_config(config).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from i6_core.util import uopen as test_func
        """
    )


def test_extra_sys_paths():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"num": 42}
    assert serialize_config(config, extra_sys_paths=[mod_path]).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        num = 42
        """
    )


def test_batch_dim():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"dim": batch_dim}
    assert serialize_config(config, inlining=False).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim as dim
        """
    )


def test_dim():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    time_dim = Dim(None, name="time")
    feat_dim = Dim(42, name="feature")
    config = {"extern_data": {"data": {"dims": [batch_dim, time_dim, feat_dim]}}}
    assert serialize_config(config, inlining=False).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim as global_batch_dim
        from returnn.tensor import Dim
        time_dim = Dim(None, name='time')
        feature_dim = Dim(42, name='feature')
        extern_data_data_dims = [global_batch_dim, time_dim, feature_dim]
        extern_data_data = {{'dims': extern_data_data_dims}}
        extern_data = {{'data': extern_data_data}}
        """
    )
    assert serialize_config(config).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim as global_batch_dim
        from returnn.tensor import Dim
        extern_data = {{'data': {{'dims': [global_batch_dim, Dim(None, name='time'), Dim(42, name='feature')]}}}}
        """
    )


def test_cached_file():
    import returnn
    from returnn.util.file_cache import CachedFile

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    cf1 = CachedFile("/path/to/some/file1.txt")
    cf2 = CachedFile("/path/to/some/file2.txt")
    config = {"obj": cf1, "obj2": cf2}
    assert serialize_config(config, inlining=False).as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from returnn.util.file_cache import CachedFile
        obj = CachedFile('/path/to/some/file1.txt')
        obj2 = CachedFile('/path/to/some/file2.txt')
        """
    )


def test_dim_hash():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    beam_dim = Dim(12, name="beam")
    config = {"beam_dim": beam_dim}
    serialized = serialize_config(config)
    assert serialized.as_serialized_code() == textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import Dim
        beam_dim = Dim(12, name='beam')
        """
    )
    coll = serialized.as_serialization_collection()
    h = sis_hash_helper(coll)
    href_ = sis_hash_helper({"delayed_objects": [("beam_dim", beam_dim)]})
    assert h == href_


def test_sis_path():
    from sisyphus import Path

    config = {"path": Path("/foo.txt")}
    assert (
        serialize_config(config, sis_path_handling=SisPathHandling.AS_STRING).as_serialized_code()
        == "path = '/foo.txt'\n"
    )


def test_post_config():
    config = {"learning_rate": 0.1}
    post_config = {"log_verbosity": 5}
    serialized = serialize_config(config, post_config)
    assert serialized.as_serialized_code() == "learning_rate = 0.1\nlog_verbosity = 5\n"
    assert len(serialized.code_list) == 2
    code1, code2 = serialized.code_list
    assert isinstance(code1, PyCode)
    assert isinstance(code2, PyCode)
    assert code1.py_name == "learning_rate" and code1.py_value_repr.py_value_repr == "0.1" and code1.use_for_hash
    assert code2.py_name == "log_verbosity" and code2.py_value_repr.py_value_repr == "5" and not code2.use_for_hash
    coll = serialized.as_serialization_collection()
    h = sis_hash_helper(coll)
    h_ref = sis_hash_helper({"delayed_objects": [("learning_rate", 0.1)]})
    assert h == h_ref


class _CustomObj:
    def __init__(self, value):
        self.value = value


def test_generic_object():
    x_orig = _CustomObj((42, (43, 44)))
    config = {"x": x_orig}
    serialized = serialize_config(config)
    print(serialized.as_serialized_code())
    # Not really checking the exact serialized code here,
    # but instead just testing to execute it.
    scope = {}
    exec(serialized.as_serialized_code(), scope)
    assert "x" in scope
    x = scope["x"]
    assert isinstance(x, _CustomObj)
    assert x.value == (42, (43, 44))
    assert x_orig is not x


class _CustomObjWithReduce:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        from copy import deepcopy

        return self.__class__, (deepcopy(self.value),)


def test_generic_object_with_reduce():
    x_orig = _CustomObjWithReduce([42, [43, 44, [45, 46]]])
    config = {"x": x_orig}
    serialized = serialize_config(config)
    print(serialized.as_serialized_code())
    # Not really checking the exact serialized code here,
    # but instead just testing to execute it.
    scope = {}
    exec(serialized.as_serialized_code(), scope)
    assert "x" in scope
    x = scope["x"]
    assert isinstance(x, _CustomObjWithReduce)
    assert x.value == [42, [43, 44, [45, 46]]]
    assert x_orig is not x


def _func(a, *, b):
    return a + b


def test_functools_partial():
    f_orig = functools.partial(_func, b=1)
    config = {"f": f_orig}
    serialized = serialize_config(config)
    print(serialized.as_serialized_code())
    # Not really checking the exact serialized code here,
    # but instead just testing to execute it.
    scope = {}
    exec(serialized.as_serialized_code(), scope)
    assert "f" in scope
    f = scope["f"]
    assert f is not f_orig
    assert isinstance(f, functools.partial)
    assert f.func is _func
    assert not f.args
    assert f.keywords == {"b": 1}
    assert f(2) == 3


def test_known_modules():
    config = {"feat_dim": Dim(12, name="feat")}
    serialized = serialize_config(config, known_modules={"returnn"})
    assert serialized.as_serialized_code() == textwrap.dedent(
        """\
        from returnn.tensor import Dim
        feat_dim = Dim(12, name='feat')
        """
    )


def test_known_module_with_conflicting_key():
    config = {"Dim": 1337, "feat_dim": Dim(12, name="feat")}
    serialized = serialize_config(config, known_modules={"returnn"})
    assert serialized.as_serialized_code() == textwrap.dedent(
        """\
        Dim = 1337
        from returnn.tensor import Dim as Dim_1
        feat_dim = Dim_1(12, name='feat')
        """
    )


def test_set():
    config = {"tags": {"a", "b", "c"}}
    serialized = serialize_config(config)
    code = serialized.as_serialized_code()
    assert code == "tags = {'a', 'b', 'c'}\n"
    scope = {}
    exec(code, scope)
    assert scope["tags"] == {"a", "b", "c"}


def test_mult_value_refs():
    config = {"a": True, "b": True}
    serialized = serialize_config(config)
    assert serialized.as_serialized_code() == textwrap.dedent(
        """\
        a = True
        b = True
        """
    )


@dataclass
class _DemoData:
    value: int


def test_dataclass():
    obj = _DemoData(42)
    config = {"obj": obj}
    serialized = serialize_config(config)
    code = serialized.as_serialized_code()
    scope = {}
    exec(code, scope)
    obj_ = scope["obj"]
    assert obj_ is not obj
    assert isinstance(obj_, _DemoData)
    assert obj_.value == 42
    assert obj_ == obj


@dataclass(frozen=True)
class _FrozenDemoData:
    value: int


def test_dataclass_frozen():
    obj = _FrozenDemoData(42)
    config = {"obj": obj}
    serialized = serialize_config(config)
    code = serialized.as_serialized_code()
    print(code)
    scope = {}
    exec(code, scope)
    obj_ = scope["obj"]
    assert obj_ is not obj
    assert isinstance(obj_, _FrozenDemoData)
    assert obj_.value == 42
    assert obj_ == obj


def test_inf():
    d = {"num": float("inf")}
    assert serialize_config(d).as_serialized_code() == "num = float('inf')\n"


@dataclass
class _DataclassWithBoundMethod:
    name: str

    def default_collect_score_results(self, x: str) -> str:
        return self.name + " " + x

    collect_score_results_func: Callable[[str], str] = None

    def __post_init__(self):
        if self.collect_score_results_func is None:
            self.collect_score_results_func = self.default_collect_score_results  # bound method


def test_bound_method():
    obj = _DataclassWithBoundMethod("foo")
    assert obj.collect_score_results_func("bar") == "foo bar"
    assert obj.collect_score_results_func.__self__ is obj
    serialized = serialize_config({"task": obj})
    code = serialized.as_serialized_code()
    print(code)
    scope = {}
    exec(code, scope)
    obj_ = scope["task"]
    assert obj_ is not obj
    assert isinstance(obj_, _DataclassWithBoundMethod)
    assert obj_.collect_score_results_func is not obj.collect_score_results_func
    assert (
        obj_.default_collect_score_results.__func__
        is obj.default_collect_score_results.__func__
        is _DataclassWithBoundMethod.default_collect_score_results
    )
    assert obj_.collect_score_results_func.__self__ is obj_
