"""
Helper code for serializing any data, e.g. for ReturnnConfig.
"""

from __future__ import annotations

import string
import sys
import textwrap
from types import FunctionType
from typing import Any, Dict, List, Optional, Tuple, Union

from i6_core.util import uopen, instanciate_delayed
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import short_hash, sis_hash_helper
from sisyphus.tools import try_get


class SerializerObject(DelayedBase):
    """
    Base class for objects that can be passed to :class:`Collection` or :class:`returnn_common.Collection`.
    """

    use_for_hash = True

    def __init__(self):
        # suppress init warning
        super().__init__(None)

    def get(self) -> str:
        """get"""
        raise NotImplementedError


class Collection(DelayedBase):
    """
    Collection of a list of :class:`SerializerObject`
    """

    def __init__(
        self,
        serializer_objects: List[SerializerObject],
    ):
        """
        :param serializer_objects: all serializer objects which are serialized into a string in order.
            For the hash, it will ignore those with use_for_hash=False.
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects

    def get(self) -> str:
        """get"""
        content = [obj.get() for obj in self.serializer_objects]
        return "".join(content)

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [obj for obj in self.serializer_objects if obj.use_for_hash],
        }
        return sis_hash_helper(h)


class Import(SerializerObject):
    """
    A class to indicate a module or function that should be imported within the returnn config

    When passed to the ReturnnCommonSerializer it will automatically detect the local package in case of
    `make_local_package_copy=True`, unless specific package paths are given.

    For imports from external libraries, e.g. git repositories use "ExternalImport".
    """

    def __init__(
        self,
        code_object_path: Union[str, FunctionType, Any],
        *,
        unhashed_package_root: Optional[str] = None,
        import_as: Optional[str] = None,
        use_for_hash: bool = True,
        ignore_import_as_for_hash: bool = False,
    ):
        """
        :param code_object_path: e.g.`i6_experiments.users.username.some_experiment.pytorch_networks.SomeNiceASRModel`.
            This can be the object itself, e.g. a function or a class. Then it will use __qualname__ and __module__.
        :param unhashed_package_root: The root path to a package, from where relatives paths will be hashed.
            Recommended is to use the root folder of an experiment module. E.g.:
            `i6_experiments.users.username.some_experiment`
            which could be retrieved via `__package__` from a module in the root of the `some_experiment` folder.
            In case one wants to avoid hash conflicts this might cause, passing an `ExplicitHash` object to the
            same collection as the import is possible.
        :param import_as: if given, the code object will be imported as this name
        :param use_for_hash: if False, this import is not hashed when passed to a Collection/Serializer
        :param ignore_import_as_for_hash: do not hash `import_as` if set
        """
        super().__init__()
        if not isinstance(code_object_path, str):
            assert getattr(code_object_path, "__qualname__", None) and getattr(code_object_path, "__module__", None)
            mod_name = code_object_path.__module__
            qual_name = code_object_path.__qualname__
            assert "." not in qual_name
            assert getattr(sys.modules[mod_name], qual_name) is code_object_path
            code_object_path = f"{mod_name}.{qual_name}"
        self.code_object = code_object_path

        self.object_name = self.code_object.split(".")[-1]
        self.module = ".".join(self.code_object.split(".")[:-1])
        self.package = ".".join(self.code_object.split(".")[:-2])

        if unhashed_package_root:
            if not self.code_object.startswith(unhashed_package_root):
                raise ValueError(
                    f"unhashed_package_root: {unhashed_package_root} is not a prefix of {self.code_object}"
                )
            self.code_object = self.code_object[len(unhashed_package_root) :]

        self.import_as = import_as
        self.use_for_hash = use_for_hash
        self.ignore_import_as_for_hash = ignore_import_as_for_hash

    def get(self) -> str:
        """get. this code is run in the task"""
        if self.import_as:
            return f"from {self.module} import {self.object_name} as {self.import_as}\n"
        return f"from {self.module} import {self.object_name}\n"

    def _sis_hash(self):
        if self.import_as and not self.ignore_import_as_for_hash:
            return sis_hash_helper({"code_object": self.code_object, "import_as": self.import_as})
        return sis_hash_helper(self.code_object)

    def __hash__(self):
        if self.import_as and not self.ignore_import_as_for_hash:
            return hash({"code_object": self.code_object, "import_as": self.import_as})
        return hash(self.code_object)


class PartialImport(Import):
    """
    Like Import, but for partial callables where certain parameters are given fixed and are hashed.
    """

    TEMPLATE = textwrap.dedent(
        """\
            ${OBJECT_NAME} = __import__("functools").partial(
                __import__("${IMPORT_PATH}", fromlist=["${IMPORT_NAME}"]).${IMPORT_NAME},
                **${KWARGS}
            )
        """
    )

    def __init__(
        self,
        *,
        code_object_path: Union[str, FunctionType, Any],
        unhashed_package_root: str,
        hashed_arguments: Dict[str, Any],
        unhashed_arguments: Dict[str, Any],
        import_as: Optional[str] = None,
        use_for_hash: bool = True,
        ignore_import_as_for_hash: bool = False,
    ):
        """
        :param code_object_path: e.g.`i6_experiments.users.username.some_experiment.pytorch_networks.SomeNiceASRModel`.
            This can be the object itself, e.g. a function or a class. Then it will use __qualname__ and __module__.
        :param unhashed_package_root: The root path to a package, from where relatives paths will be hashed.
            Recommended is to use the root folder of an experiment module. E.g.:
            `i6_experiments.users.username.some_experiment`
            which could be retrieved via `__package__` from a module in the root of the `some_experiment` folder.
            In case one wants to avoid hash conflicts this might cause, passing an `ExplicitHash` object to the
            same collection as the import is possible.
        :param hashed_arguments: argument dictionary for addition partial arguments to set to the callable.
            Will be serialized as dict into the config, so make sure to use only serializable/parseable content
        :param unhashed_arguments: same as above, but does not influence the hash
        :param import_as: if given, the code object will be imported as this name
        :param use_for_hash: if False, this module is not hashed when passed to a Collection/Serializer
        :param ignore_import_as_for_hash: do not hash `import_as` if set
        """

        super().__init__(
            code_object_path=code_object_path,
            unhashed_package_root=unhashed_package_root,
            import_as=import_as,
            use_for_hash=use_for_hash,
            ignore_import_as_for_hash=ignore_import_as_for_hash,
        )
        self.hashed_arguments = hashed_arguments
        self.unhashed_arguments = unhashed_arguments

    def get(self) -> str:
        arguments = {**self.unhashed_arguments, **self.hashed_arguments}
        return string.Template(self.TEMPLATE).substitute(
            {
                "KWARGS": str(instanciate_delayed(arguments)),
                "IMPORT_PATH": self.module,
                "IMPORT_NAME": self.object_name,
                "OBJECT_NAME": self.import_as if self.import_as is not None else self.object_name,
            }
        )

    def _sis_hash(self):
        super_hash = super()._sis_hash()
        return sis_hash_helper({"import": super_hash, "hashed_arguments": self.hashed_arguments})


class ExternalImport(SerializerObject):
    """
    Import from e.g. a git repository. For imports within the recipes use "Import".

    Should be added in the beginning.
    """

    def __init__(self, import_path: tk.Path):
        super().__init__()
        self.import_path = import_path

    def get(self) -> str:
        return f'sys.path.insert(0, "{self.import_path.get()}")\n'

    def _sis_hash(self):
        return sis_hash_helper(self.import_path)


class CodeFromFunction(SerializerObject):
    """
    Insert code from function.
    """

    def __init__(self, name: str, func: FunctionType, *, hash_full_python_code: bool = False):
        """
        :param name: name of the function as exposed in the config
        :param func:
        :param hash_full_python_code: if True, the full python code of the function is hashed,
            otherwise only the module name and function qualname are hashed.
        """
        super().__init__()
        self.name = name
        self.func = func
        self.hash_full_python_code = hash_full_python_code

        # Similar as ReturnnConfig.
        import inspect

        self._func_code = inspect.getsource(self.func)
        code_hash = short_hash(self._func_code)
        if self.func.__name__ == self.name:
            self._code = self._func_code
        else:
            # Wrap the code inside a function to be sure we do not conflict with other names.
            self._code = "".join(
                [
                    f"def _{self.name}_{code_hash}():\n",
                    textwrap.indent(self._func_code, "    "),
                    "\n",
                    f"    return {self.func.__name__}\n",
                    f"{self.name} = _{self.name}_{code_hash}()\n",
                ]
            )

    def get(self):
        """get"""
        return self._code

    def _sis_hash(self):
        if self.hash_full_python_code:
            return sis_hash_helper((self.name, self._func_code))
        else:
            return sis_hash_helper((self.name, f"{self.func.__module__}.{self.func.__qualname__}"))


# noinspection PyAbstractClass
class _NonhashedSerializerObject(SerializerObject):
    """
    Any serializer object which is not used for the hash.
    """

    use_for_hash = False

    def _sis_hash(self):
        raise Exception(f"{self.__class__.__name__} must not be hashed")


class NonhashedCode(_NonhashedSerializerObject):
    """
    Insert code from raw string which is not hashed.
    """

    def __init__(self, code: Union[str, tk.Path]):
        super().__init__()
        self.code = code

    def get(self):
        """get"""
        return self.code


class NonhashedCodeFromFile(_NonhashedSerializerObject):
    """
    Insert code from file content which is not hashed (neither the file name nor the content).
    """

    def __init__(self, filename: tk.Path):
        super().__init__()
        self.filename = filename

    def get(self):
        """get"""
        with uopen(self.filename, "rt") as f:
            return f.read()


class CodeFromFile(SerializerObject):
    """
    Insert code from a file hashed by file path/name or full content
    """

    def __init__(self, filename: tk.Path, hash_full_content: bool = False):
        """
        :param filename:
        :param hash_full_content: False -> hash filename, True -> hash content (but not filename)
        """
        super().__init__()
        self.filename = filename
        self.hash_full_content = hash_full_content

    def get(self):
        """get"""
        with uopen(self.filename, "rt") as f:
            return f.read()

    def _sis_hash(self):
        if self.hash_full_content:
            with uopen(self.filename, "rt") as f:
                return sis_hash_helper(f.read())
        else:
            return sis_hash_helper(self.filename)


class ExplicitHash(SerializerObject):
    """
    Inserts nothing, but uses the given object for hashing
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, hash: Any):
        super().__init__()
        self.hash = hash

    def get(self) -> str:
        """get"""
        return ""

    def _sis_hash(self):
        return sis_hash_helper(self.hash)


class Call(SerializerObject):
    """
    SerializerObject that serializes the call of a callable with given arguments.
    The return values of the call are optionally assigned to variables of a given name.
    Example:
    Call(callable_name="range", kwargs=[("start", 1), ("stop", 10)], return_assign_variables="number_range")
    ->
    number_range = range(start=1, stop=10)
    """

    def __init__(
        self,
        callable_name: str,
        kwargs: Optional[List[Tuple[str, Union[str, DelayedBase]]]] = None,
        unhashed_kwargs: Optional[List[Tuple[str, Union[str, DelayedBase]]]] = None,
        return_assign_variables: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        :param callable_name: Name of the callable for which the call is serialized.
        :param args: Optional list of positional arguments provided to the call.
        :param kwargs: Optional list of keyword arguments provided to the call in the form of key-value tuples.
        :param return_assign_variables: Optional name or list of variable names that the return value(s) of the call are assigned to.
        """
        self.callable_name = callable_name
        self.kwargs = kwargs or []
        self.unhashed_kwargs = unhashed_kwargs or []
        self.return_assign_variables = return_assign_variables

        if isinstance(self.return_assign_variables, str):
            self.return_assign_variables = [self.return_assign_variables]

    def get(self) -> str:
        # Variable assignment
        return_assign_str = ""
        if self.return_assign_variables is not None:
            return_assign_str = ", ".join(self.return_assign_variables) + " = "

        # kwargs
        kwargs_str_list = [f"{key}={try_get(val)}" for key, val in self.kwargs + self.unhashed_kwargs]

        # full call
        return f"{return_assign_str}{self.callable_name}({', '.join(kwargs_str_list)})"

    def _sis_hash(self):
        h = {
            "callable_name": self.callable_name,
            "kwargs": self.kwargs,
            "return_assign_variables": self.return_assign_variables,
        }
        return sis_hash_helper(h)


PythonEnlargeStackWorkaroundNonhashedCode = NonhashedCode(
    textwrap.dedent(
        """\
        # https://github.com/rwth-i6/returnn/issues/957
        # https://stackoverflow.com/a/16248113/133374
        import resource
        import sys
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
        except Exception as exc:
            print(f"resource.setrlimit {type(exc).__name__}: {exc}")
        sys.setrecursionlimit(10 ** 6)
        """
    )
)

PythonCacheManagerFunctionNonhashedCode = NonhashedCode(
    textwrap.dedent(
        """\
        _cf_cache = {}

        def cf(filename):
            "Cache manager"
            from subprocess import check_output, CalledProcessError
            if filename in _cf_cache:
                return _cf_cache[filename]
            if int(os.environ.get("RETURNN_DEBUG", "0")):
                print("use local file: %s" % filename)
                return filename  # for debugging
            try:
                cached_fn = check_output(["cf", filename]).strip().decode("utf8")
            except CalledProcessError:
                print("Cache manager: Error occurred, using local file")
                return filename
            assert os.path.exists(cached_fn)
            _cf_cache[filename] = cached_fn
            return cached_fn
        """
    )
)

# Modelines should be at the beginning or end of a file.
# Many editors (e.g. VSCode) read those information.
PythonModelineNonhashedCode = NonhashedCode("# -*- mode: python; tab-width: 4 -*-\n")
