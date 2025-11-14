"""
New simplified config serialization, usually for RETURNN configs.

Allows serializing arbitrary Python objects inside the config
including functions/classes/modules/objects,
with automatic import handling.

See https://github.com/rwth-i6/i6_experiments/blob/main/users/zeyer/serialization.rst
for more details.

This is conceptually similar to :class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`
and :func:`i6_experiments.common.setups.returnn.serialization.get_serializable_config`.

See :func:`serialize_config` for the main entry point.
See :class:`ReturnnConfigWithNewSerialization` for an easy :class:`ReturnnTrainingJob` integration.

Note: Sisyphus hashes are currently just defined by the config keys/values,
using the `sis_hash_helper` function, without any special handling,
except for dim tags (RETURNN :class:`Dim` objects).
That means, e.g. functions/classes get hashed by ``(obj.__module__, obj.__qualname__)``.
We currently don't provide a way to customize the hashing behavior
(except of ``post_config`` which is not hashed at all).
This could be extended in the future by allowing more custom behavior
e.g. for module scopes.
Also, e.g. to specify ``unhashed_package_root`` for some of the references.

Note: Sisyphus Path objects are serialized directly using :func:`sisyphus.Path.get_path`.

We handle those objects specially:
- primitive types (int, float, bool, str)
- Sisyphus Path objects
- RETURNN Dim and CachedFile objects
- dict, list, tuple, set
- functions, classes, modules
- functools.partial (just some nicer repr)
- i6_core.serialization objects
- CodeWrapper

All other generic objects are handled in the same way as pickle does it
(or also :class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`),
i.e. using ``__reduce__`` (etc).
This also allows circular references.
"""

from __future__ import annotations


__all__ = [
    "serialize_config",
    "SisPathHandling",
    "ReturnnConfigWithNewSerialization",
    "SerializedConfig",
    "SerializationError",
    "PyCode",
    "PyEvalCode",
]

import builtins
from dataclasses import dataclass
import enum
import functools
from keyword import iskeyword
import math
import os
import re
import subprocess
import sys
import types
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from sisyphus import Path
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

from i6_core.returnn.config import CodeWrapper, ReturnnConfig, unparse_python
from i6_core.returnn.training import Checkpoint, PtCheckpoint
from i6_core.serialization import (
    Call,
    CallImport,
    CodeFromFile,
    CodeFromFunction,
    ExplicitHash,
    ExternalImport,
    Import,
    NonhashedCode,
    PartialImport,
    SerializerObject,
)
from i6_core.serialization import Collection as SerializerCollection

if TYPE_CHECKING:
    from returnn.tensor import Dim
    from returnn.util.file_cache import CachedFile


def serialize_config(
    config: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    *,
    inlining: bool = True,
    known_modules: Collection[str] = (),
    extra_sys_paths: Sequence[str] = (),
    sis_path_handling: SisPathHandling = None,
) -> SerializedConfig:
    """serialize config. see module docstring for more info."""
    serializer = _Serializer(
        config=config, post_config=post_config, known_modules=known_modules, sis_path_handling=sis_path_handling
    )
    for path in extra_sys_paths:
        serializer.add_sys_path(path, recursive=False)
    serializer.work_queue()
    if inlining:
        serializer.work_inlining()
    return SerializedConfig(code_list=list(serializer.assignments_dict_by_idx.values()))


class SisPathHandling(enum.Enum):
    """
    SisPathHandling enum.
    """

    NONE = None
    AS_STRING = "as_string"
    NO_DEPS = "no_deps"


def _instanciate_delayed_copy(o: Any) -> Any:
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    In contrast to :func:`i6_core.util.instanciate_delayed` this function does not operate inplace.

    :param o: nested structure that may contain DelayedBase objects
    :return: o with all DelayedBase objects replaced by their .get() value
    """
    import tree

    def _instanciate_delayed_obj(o: Any) -> Any:
        if isinstance(o, DelayedBase) and not isinstance(o, SerializerObject):
            return o.get()
        return o

    return tree.map_structure(_instanciate_delayed_obj, o)


def _is_valid_python_identifier_name(name: str) -> bool:
    """
    :return: whether the name is a valid Python identifier name (including attrib name)
    """
    return name.isidentifier() and not iskeyword(name)


class ReturnnConfigWithNewSerialization(ReturnnConfig):
    """
    Overwrites the serialization behavior of ReturnnConfig.
    """

    @staticmethod
    def from_cfg(old_returnn_cfg: ReturnnConfig):
        """
        Creates a ReturnnConfigWithNewSerialization from an existing ReturnnConfig.

        This is used to override the serialization behavior to V2.
        """

        assert not old_returnn_cfg.staged_network_dict, "V2 serialization does not support staged net dicts"
        return ReturnnConfigWithNewSerialization(
            config=old_returnn_cfg.config,
            hash_full_python_code=old_returnn_cfg.hash_full_python_code,
            post_config=old_returnn_cfg.post_config,
            python_epilog=old_returnn_cfg.python_epilog,
            python_epilog_hash=old_returnn_cfg.python_epilog_hash,
            python_prolog=old_returnn_cfg.python_prolog,
            python_prolog_hash=old_returnn_cfg.python_prolog_hash,
            sort_config=False,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _serialize(self) -> str:
        import tree

        # This is usually run within the worker, but it shouldn't really matter.
        assert not self.staged_network_dict  # not supported

        self.check_consistency()

        config = _instanciate_delayed_copy(self.config)
        post_config = _instanciate_delayed_copy(self.post_config)

        # I'm not really sure about it.
        # Our automatic mechanism will find direct imports (e.g. i6_experiments).
        # However, it will not find indirect imports (e.g. sisyphus),
        # and thus the generated code might fail.
        # So add all other paths here which we currently have.
        # (While testing this, this was sisyphus + returnn + recipes,
        #  but returnn is excluded below.)
        sys_paths = set(_get_base_sys_path_list())
        extra_sys_paths = [p for p in sys.path if p not in sys_paths]

        # Handle ExternalImports
        extra_sys_paths += [
            item.import_path
            for item in tree.flatten(config) + tree.flatten(post_config)
            if isinstance(item, ExternalImport)
        ]

        python_prolog_code = unparse_python(self.python_prolog, hash_full_python_code=True)
        python_epilog_code = unparse_python(self.python_epilog, hash_full_python_code=True)
        serialized = serialize_config(
            config,
            post_config,
            # Of course RETURNN knows about itself, no need to add to sys.path.
            # Also, we don't want to force the current RETURNN here,
            # but allow the config to be used with any other RETURNN version.
            known_modules={"returnn"},
            extra_sys_paths=extra_sys_paths,
            # instanciate_delayed should already have handled it (e.g. Path),
            # or if it has not, then we want to keep it as it is (e.g. PtCheckpoint),
            # but without dependencies.
            sis_path_handling=SisPathHandling.AS_STRING,
        )
        return "\n\n".join(
            part
            for part in [
                "#!returnn/rnn.py",
                python_prolog_code,
                serialized.as_serialized_code(),
                python_epilog_code,
                "# -*- mode: python; tab-width: 4 -*-\n",
            ]
            if part.strip()
        )


@dataclass
class SerializedConfig:
    code_list: List[PyCode]

    def as_serialization_collection(self) -> SerializerCollection:
        """as serialization Collection"""
        return SerializerCollection(self.code_list)

    def as_serialized_code(self) -> str:
        """as serialized code"""
        return "".join(code.py_code for code in self.code_list)


class _Serializer:
    def __init__(
        self,
        config: Dict[str, Any],
        post_config: Optional[Dict[str, Any]] = None,
        known_modules: Collection[str] = (),
        sis_path_handling: SisPathHandling = None,
    ):
        self.config = config.copy()
        self.post_config = post_config.copy() if post_config else {}
        self.assignments_dict_by_value_ref: Dict[_Ref, PyCode] = {}  # value ref -> code
        self.assignments_dict_by_name: Dict[str, PyCode] = {}  # var name -> code
        self.assignments_dict_by_idx: Dict[int, PyCode] = {}  # idx -> code
        self.assignments_dict_by_value_by_type: Dict[type, Dict[Any, PyCode]] = {
            # TODO: is this entry needed for correctness?
            # Dim: {},
        }  # type -> dict value -> code
        self.reduce_cache_by_value_ref: Dict[_Ref, Tuple[Any, ...]] = {}  # value ref -> (func, args, ...)
        self.added_sys_paths = set()
        self.known_modules = set(known_modules)
        for mod_name in known_modules:
            mod = sys.modules[mod_name]
            # Don't add those module path to sys.path again.
            self.added_sys_paths.add(_get_module_path_from_module(mod))
        self._next_sys_path_insert_idx = 0
        self.sis_path_handling = sis_path_handling
        self._cur_added_refs: List[PyCode] = []
        self._next_assignment_idx = 0
        # We first serialize everything without inlining anything.
        # There we also count how often a value is used (ref_count).
        # Then we can inline those values which are not direct config entries
        # and which are only used once.
        self._inlining_stage = False

        # Avoid to use them, but if necessary (when inside the config), they can be used.
        # The config keys always have precedence.
        self._internal_reserved_names = {"sys": sys}
        self._internal_reserved_names_by_value_ref = {
            _Ref(value): name for name, value in self._internal_reserved_names.items()
        }

    def work_queue(self):
        """
        Runs the serialization queue.

        Conceptually this runs a depth-first search over the config dict, writing
        values into the config as it finds them.

        Whenever it encounters a value that needs to be serialized before the
        current/original value, an exception is raised with the dependency value to be
        serialized next. This value is then added to the front of the queue, and the
        current item is left in place.

        The dependency is then serialized and a reference to it is kept in
        `assignments_dict_by_value_ref`. When we re-try serializing the original
        value, we again attempt the recursive serialization of the dependency, but
        this time we find it in `assignments_dict_by_value_ref` and so we can use it
        directly and proceed serializing the (original) value.

        Eventually the queue runs empty and we are done with the serialization.
        At this point, `work_inlining` can be run to inline the values that are
        only used once and are not direct config entries.

        The resulting code can then be fetched from `assignments_dict_by_idx`.
        """

        self._inlining_stage = False
        queue: List[Union[_AssignQueueItem, _DeferredStateQueueItem]] = [
            _AssignQueueItem(required_var_name=key, value=value)
            for key, value in list(self.config.items()) + list(self.post_config.items())
        ]
        queue.reverse()  # we will pop from the end
        while queue:
            deferred_state: Optional[_DeferredStateQueueItem] = None
            queue_item = queue[-1]
            self._cur_added_refs.clear()
            try:
                if isinstance(queue_item, _AssignQueueItem):
                    deferred_state = self._handle_next_queue_item(queue_item)
                elif isinstance(queue_item, _DeferredStateQueueItem):
                    self._handle_deferred_state_queue_item(queue_item)
                else:
                    raise TypeError(f"unexpected queue item type {type(queue_item).__name__}")
                assert queue[-1] is queue_item
                queue.pop(-1)
            except _SerializationDependsOnNotYetSerializedOtherVarException as exc:
                exc.queue_item.via_queue_item = queue_item
                queue.append(exc.queue_item)
                for code in self._cur_added_refs:
                    code.ref_count -= 1
            if deferred_state:
                queue.append(deferred_state)

    def work_inlining(self):
        self._inlining_stage = True
        self._next_assignment_idx = -1
        for assign in list(self.assignments_dict_by_idx.values()):
            assert assign.idx > self._next_assignment_idx
            self._next_assignment_idx = assign.idx
            if assign.py_name and not assign.has_later_state_setup:
                new_assign = self._serialize_value_assignment(assign.value, name=assign.py_name)
                assert isinstance(new_assign, PyCode)
                assign.py_value_repr = new_assign.py_value_repr
                assign.py_code = new_assign.py_code
        self._next_assignment_idx += 1

    def _handle_next_queue_item(self, queue_item: _AssignQueueItem) -> Optional[_DeferredStateQueueItem]:
        value_ref = _Ref(queue_item.value)
        if queue_item.required_var_name:
            assert queue_item.required_var_name not in self.assignments_dict_by_name
        if not queue_item.required_var_name and value_ref in self.assignments_dict_by_value_ref:
            # No need to assign it again.
            return None
        name = queue_item.required_var_name
        if not name and value_ref in self._internal_reserved_names_by_value_ref:
            name = self._get_unique_suggested_name(
                self._internal_reserved_names_by_value_ref[value_ref], allow_internal_reserved_name=True
            )
        if not name and (
            isinstance(queue_item.value, (type, FunctionType, BuiltinFunctionType, ModuleType, Import, Call))
            or (getattr(queue_item.value, "__module__", None) and getattr(queue_item.value, "__qualname__", None))
            or (_isinstance_returnn_dim(queue_item.value) and queue_item.value.name)
        ):
            # For those types, prefer a name based on the value, even over any other suggested name.
            name = self._get_unique_suggested_name(self._suggest_name_from_value(queue_item.value))
        if not name and queue_item.suggested_var_name:
            name = self._get_unique_suggested_name(queue_item.suggested_var_name)
        if not name:
            name = self._get_unique_suggested_name(self._suggest_name_from_value(queue_item.value))
        serialized = self._serialize_value_assignment(value=queue_item.value, name=name)
        deferred_state: Optional[_DeferredStateQueueItem] = None
        if isinstance(serialized, _PyCodeWithDeferredStateQueueItem):
            serialized, deferred_state = serialized.code, serialized.extra
            serialized.has_later_state_setup = True
            deferred_state.via_queue_item = queue_item
        assert isinstance(serialized, PyCode)
        serialized.idx = self._next_assignment_idx
        self._next_assignment_idx += 1
        if queue_item.required_var_name:
            serialized.is_direct_config_entry = True
            if queue_item.required_var_name in self.config:
                serialized.use_for_hash = True
        assert serialized.py_name == name
        assert name not in self.assignments_dict_by_name  # double check
        self.assignments_dict_by_name[name] = serialized
        if value_ref not in self.assignments_dict_by_value_ref:
            self.assignments_dict_by_value_ref[value_ref] = serialized
        value_dict = self.assignments_dict_by_value_by_type.get(type(queue_item.value))
        if value_dict is not None:
            if queue_item.value in value_dict:
                if serialized.is_direct_config_entry:
                    # Same reasoning as above for assignments_dict_by_value_ref.
                    assert value_dict[queue_item.value].is_direct_config_entry
            else:
                value_dict[queue_item.value] = serialized
        self.assignments_dict_by_idx[serialized.idx] = serialized
        return deferred_state

    def _handle_deferred_state_queue_item(self, rv: _DeferredStateQueueItem):
        name = rv.py_name
        value = rv.value
        state = rv.state
        listitems = rv.listitems
        dictitems = rv.dictitems
        state_setter = rv.state_setter
        code_lines = []

        if listitems is not None:
            for i, item in enumerate(listitems):
                item_s = self._serialize_value(item, prefix=f"{name}_listitem{i}", recursive=True)
                code_lines.append(f"{name}.append({item_s.py_inline()})\n")

        if dictitems is not None:
            for key, v in dictitems:
                serialized_key = self._serialize_value(key, prefix=f"{name}_key", recursive=True)
                assert isinstance(serialized_key, PyEvalCode)
                if (isinstance(key, str) and _is_valid_python_identifier_name(key)) or isinstance(key, (int, bool)):
                    prefix_name = str(key)
                else:
                    prefix_name = "value"
                serialized_value = self._serialize_value(v, prefix=f"{name}_{prefix_name}", recursive=True)
                assert isinstance(serialized_value, PyEvalCode)
                code_lines.append(f"{name}[{serialized_key.py_inline()}] = {serialized_value.py_inline()}\n")

        if state is not None:
            if state_setter is None:
                # See pickle._Unpickler.load_build.
                setstate = getattr(value, "__setstate__", None)
                if setstate is not None:
                    state_s = self._serialize_value(state, prefix=f"{name}_state", recursive=True)
                    assert isinstance(state_s, PyEvalCode)
                    code_lines.append(f"{name}.__setstate__({state_s.py_inline()})\n")
                else:
                    slotstate = None
                    if isinstance(state, tuple) and len(state) == 2:
                        state, slotstate = state
                    if state:
                        state_s = self._serialize_value(state, prefix=f"{name}_state", recursive=True)
                        assert isinstance(state_s, PyEvalCode)
                        code_lines.append(f"{name}.__dict__.update({state_s.py_inline()})\n")
                    if slotstate:
                        # not handled yet
                        raise NotImplementedError(
                            f"serialize {rv.py_name} = {rv.value!r} with slotstate {slotstate!r},"
                            f" via {rv.debug_trace()}"
                        )

            else:
                raise NotImplementedError  # not handled yet

        code = PyCode(py_name=None, value=None, py_code="".join(code_lines))
        code.idx = self._next_assignment_idx
        self._next_assignment_idx += 1
        self.assignments_dict_by_idx[code.idx] = code

    @staticmethod
    def _suggest_name_from_value(value: Any) -> str:
        if _isinstance_returnn_dim(value):
            return _Serializer._suggested_name_for_dim(value)
        if isinstance(value, (Import, PartialImport, CallImport)):
            return value.import_as or value.object_name
        if isinstance(value, Call) and value.return_assign_variables is not None:
            if isinstance(value.return_assign_variables, str):
                return value.return_assign_variables
            elif isinstance(value.return_assign_variables, list) and len(value.return_assign_variables) == 1:
                return value.return_assign_variables[0]
            else:
                raise NotImplementedError(
                    "Cannot destructure-assign multiple return variables from calls in serialization_v2"
                )
        if isinstance(value, CodeFromFunction):
            return value.name
        if isinstance(value, (type, ModuleType, FunctionType, BuiltinFunctionType)) and getattr(
            value, "__qualname__", None
        ):
            return value.__qualname__.replace(".", "_")
        if getattr(value, "__module__", None) and getattr(value, "__qualname__", None):
            return f"{value.__module__}.{value.__qualname__}".replace(".", "_")
        if getattr(value, "__qualname__", None):
            return value.__qualname__.replace(".", "_")
        if getattr(value, "__name__", None):
            return value.__name__
        return type(value).__name__.lower()

    @staticmethod
    def _suggested_name_for_dim(dim: Dim) -> str:
        if not dim.name:
            return "dim"  # fallback
        name_ = dim.name
        name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
        if not name_:
            return "dim"  # fallback
        if name_[:1].isdigit():
            return "dim_" + name_
        if not name_.endswith("_dim"):
            name_ += "_dim"
        return name_

    def _get_unique_suggested_name(self, suggested_name: str, *, allow_internal_reserved_name: bool = False) -> str:
        # If we ever get here and the suggested name is not a valid Python identifier,
        # then we can sanitize it here.
        assert _is_valid_python_identifier_name(suggested_name)  # not handled yet otherwise...
        if self._check_can_use_suggested_name(
            suggested_name, allow_internal_reserved_name=allow_internal_reserved_name
        ):
            return suggested_name
        i = 1
        while True:
            name = f"{suggested_name}_{i}"
            if self._check_can_use_suggested_name(name, allow_internal_reserved_name=allow_internal_reserved_name):
                return name
            i += 1

    def _check_can_use_suggested_name(self, name: str, *, allow_internal_reserved_name: bool = False) -> bool:
        if not allow_internal_reserved_name and name in self._internal_reserved_names:
            return False
        if name in builtins.__dict__:  # e.g. `len`, `sum`, etc.
            return False
        if name in self.config:
            return False
        if name in self.assignments_dict_by_name:
            return False
        return True

    def _serialize_value_assignment(self, value: Any, name: str) -> Union[PyCode, _PyCodeWithDeferredStateQueueItem]:
        serialized = self._serialize_value(value=value, prefix=name, recursive=False, name=name)
        if isinstance(serialized, PyEvalCode):
            return PyCode(
                py_name=name,
                value=value,
                py_code=f"{name} = {serialized.py_value_repr}\n",
                py_value_repr=serialized,
            )
        elif isinstance(serialized, (PyCode, _PyCodeWithDeferredStateQueueItem)):
            return serialized
        else:
            raise TypeError(f"unexpected serialized type {type(serialized).__name__}")

    def _serialize_value(
        self, value: Any, prefix: str, *, recursive: bool = True, name: Optional[str] = None
    ) -> Union[PyEvalCode, PyCode, _PyCodeWithDeferredStateQueueItem]:
        # The code here is somewhat similar as pickle._Pickler.save,
        # but we have some special treatment for a few types.
        value_ref = _Ref(value)
        if value is None:
            return PyEvalCode("None")
        if isinstance(value, (int, float, bool, str, bytes)):
            if isinstance(value, float) and not math.isfinite(value):
                return PyEvalCode(f"float('{value}')")
            return PyEvalCode(repr(value))
        if self.sis_path_handling and isinstance(value, Path):
            return self._serialize_sis_path(value)
        if isinstance(value, CodeWrapper):
            return self._serialize_code_wrapper(value)
        if getattr(value, "__module__", None) == "builtins":
            val_name: str = getattr(value, "__name__", None)
            if val_name and getattr(builtins, val_name, None) is value:
                assign = self.assignments_dict_by_name.get(val_name)
                if not assign or assign.idx >= self._next_assignment_idx:
                    return PyEvalCode(val_name)
                # name was overwritten. fallback to standard module access.
        # Note that assignments_dict_by_value_ref would also contain primitive objects like True/False etc.
        # But most of those are handled above already, so we would not reuse them here.
        if value_ref in self.assignments_dict_by_value_ref:
            assign = self.assignments_dict_by_value_ref[value_ref]
            if self._inlining_stage:
                if assign.idx >= self._next_assignment_idx:
                    pass  # self, or future ref, cannot use this, proceed serializing
                elif assign.is_direct_config_entry:
                    return PyEvalCode(assign.py_name)  # anyway need to keep this assignment, so just use it
                else:
                    assert assign.ref_count >= 1
                    if assign.ref_count > 1:
                        # there are multiple references, so we need to keep this assignment
                        return PyEvalCode(assign.py_name)
                    if not assign.py_value_repr:
                        return PyEvalCode(assign.py_name)  # we cannot inline this, so just use the assignment
                    # We can inline this.
                    # Thus remove the reference to this assignment.
                    assign.ref_count -= 1
                    assert assign.ref_count == 0
                    # Can delete this assignment.
                    del self.assignments_dict_by_value_ref[value_ref]
                    del self.assignments_dict_by_name[assign.py_name]
                    del self.assignments_dict_by_idx[assign.idx]
                    return assign.py_value_repr
            else:
                assign.ref_count += 1
                self._cur_added_refs.append(assign)
                return PyEvalCode(assign.py_name)
        if not self._inlining_stage:
            value_dict = self.assignments_dict_by_value_by_type.get(type(value))
            if value_dict is not None and value in value_dict:
                assign = value_dict.get(value)
                if assign is not None:
                    assign.ref_count += 1
                    self._cur_added_refs.append(assign)
                    return PyEvalCode(assign.py_name)

        # Any of the following could potentially cause further recursive calls,
        # thus check the recursive flag at this point.
        if recursive:
            assert not self._inlining_stage  # should not get here when inlining
            raise _SerializationDependsOnNotYetSerializedOtherVarException(
                _AssignQueueItem(value=value, suggested_var_name=prefix),
            )
        assert name

        if isinstance(value, dict):
            return self._serialize_dict(value, prefix)
        if isinstance(value, list):
            return self._serialize_list(value, prefix)
        if isinstance(value, tuple):
            return self._serialize_tuple(value, prefix)
        if isinstance(value, set):
            return self._serialize_set(value, prefix)
        if _isinstance_returnn_dim(value):
            return self._serialize_dim(value, prefix)
        if _isinstance_returnn_cached_file(value):
            return self._serialize_cached_file(value)
        if isinstance(value, functools.partial):
            return self._serialize_functools_partial(value, name)
        if isinstance(value, Call):
            return self._serialize_call(value, name)
        if isinstance(value, CallImport):
            return self._serialize_call_import(value, name)
        if isinstance(value, PartialImport):
            return self._serialize_partial_import(value, name)
        if isinstance(value, Import):
            return self._serialize_import(value, name)
        if isinstance(value, (Checkpoint, PtCheckpoint)):
            return self._serialize_checkpoint(value)
        if isinstance(value, CodeFromFunction):
            return self._serialize_code_from_function(value, name)
        if isinstance(value, (CodeFromFile, ExplicitHash, ExternalImport, NonhashedCode)):
            raise ValueError(
                f"Cannot serialize {type(value).__name__} in config dict. "
                "It does not represent a value and should go into python_prolog/python_epilog."
            )
        if isinstance(value, ModuleType):
            return self._serialize_module(value, name)
        if isinstance(value, MethodType):
            return self._serialize_method(value, name)

        if isinstance(value, (type, FunctionType, BuiltinFunctionType)) or (
            getattr(value, "__module__", None) and getattr(value, "__qualname__", None)
        ):
            return self._serialize_global(value=value, name=name)

        # Generic fallback using __reduce__ or __reduce_ex__.
        return self._serialize_reduce(value, name)

    def _serialize_reduce(self, value: Any, name: str) -> Union[PyEvalCode, _PyCodeWithDeferredStateQueueItem]:
        # Generic fallback using __reduce__ or __reduce_ex__.
        # This is very much following the original pickle logic (slightly simplified).
        value_ref = _Ref(value)
        if value_ref in self.reduce_cache_by_value_ref:
            rv = self.reduce_cache_by_value_ref[value_ref]
        else:
            reduce = getattr(value, "__reduce_ex__", None)
            reduce_proto = 4  # not sure...
            if reduce is not None:
                rv = reduce(reduce_proto)
            else:
                reduce = getattr(value, "__reduce__", None)
                if reduce is not None:
                    rv = reduce()
                else:
                    assert not self._inlining_stage  # should really not happen in this stage
                    raise SerializationError(
                        f"cannot handle `({name}) = {value!r}` (value type {type(value).__name__})"
                    )

            # Check for string returned by reduce(), meaning "save as global"
            if isinstance(rv, str):
                return self._serialize_global(value, name=rv)

            # Assert that reduce() returned a tuple
            if not isinstance(rv, tuple):
                raise SerializationError(f"{reduce} must return string or tuple, got {rv!r} (type {type(rv).__name__})")

            # Assert that it returned an appropriately sized tuple
            if not (2 <= len(rv) <= 6):
                raise SerializationError(f"Tuple returned by {reduce} invalid num elements {len(rv)}: {rv!r}")

            # Keep it cached, such that we do not re-execute the `reduce` again,
            # which might created new temporary objects on-the-fly,
            # thus our recursive construction does not work.
            self.reduce_cache_by_value_ref[value_ref] = rv

        # func, args, state=None, listitems=None, dictitems=None, state_setter=None
        func, args = rv[:2]
        func_s = self._serialize_value(func, prefix=f"{name}_reduce_func", recursive=True)
        assert isinstance(func_s, PyEvalCode)
        assert isinstance(args, (tuple, list))
        args_s = [
            self._serialize_value(arg, prefix=f"{name}_reduce_arg{i}", recursive=True) for i, arg in enumerate(args)
        ]
        assert all(isinstance(a, PyEvalCode) for a in args_s)
        code_s = func_s.py_inline() + "(" + ", ".join(arg_s.py_inline() for arg_s in args_s) + ")"
        if len(rv) == 2:
            return PyEvalCode(code_s)
        return _PyCodeWithDeferredStateQueueItem(
            code=PyCode(py_name=name, value=value, py_code=f"{name} = {code_s}\n", has_later_state_setup=True),
            extra=_DeferredStateQueueItem(name, value, *rv[2:]),
        )

    def _serialize_dict(self, values: dict, prefix: str) -> PyEvalCode:
        # nothing else expected/handled currently, isinstance is wrong
        assert type(values) is dict  # noqa
        serialized_items = []
        for key, value in values.items():
            serialized_key = self._serialize_value(key, prefix=f"{prefix}_key", recursive=True)
            assert isinstance(serialized_key, PyEvalCode)
            if (isinstance(key, str) and _is_valid_python_identifier_name(key)) or isinstance(key, (int, bool)):
                prefix_name = str(key)
            else:
                prefix_name = "value"
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{prefix_name}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(f"{serialized_key.py_inline()}: {serialized_value.py_inline()}")
        return PyEvalCode("{" + ", ".join(serialized_items) + "}")

    def _serialize_list(self, values: list, prefix: str) -> PyEvalCode:
        # nothing else expected/handled currently
        assert type(values) is list  # noqa
        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(serialized_value.py_inline())
        return PyEvalCode("[" + ", ".join(serialized_items) + "]")

    def _serialize_tuple(self, values: tuple, prefix: str) -> PyEvalCode:
        if not values:
            if type(values) is tuple:
                return PyEvalCode("()")
            # Assume namedtuple.
            type_s = self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True)
            assert isinstance(type_s, PyEvalCode)
            return PyEvalCode(f"{type_s.py_inline()}()")

        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(serialized_value.py_inline())

        if type(values) is tuple:
            return PyEvalCode("(" + ", ".join(serialized_items) + (")" if len(values) > 1 else ",)"))
        # Assume namedtuple.
        # noinspection PyUnresolvedReferences,PyProtectedMember
        fields = values._fields
        assert len(fields) == len(serialized_items)
        value_type_str = self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True)
        assert isinstance(value_type_str, PyEvalCode)
        return PyEvalCode(
            f"{value_type_str.py_inline()}("
            + ", ".join(f"{key}={value}" for key, value in zip(fields, serialized_items))
            + ")"
        )

    def _serialize_set(self, values: set, prefix: str) -> PyEvalCode:
        # nothing else expected/handled currently
        assert type(values) is set  # noqa
        if not values:
            assert "set" not in self.assignments_dict_by_name  # just not yet handled...
            return PyEvalCode("set()")
        values = list(values)
        # noinspection PyBroadException
        try:
            values.sort()
        except Exception:
            pass  # ignore sort errors, not critical
        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(serialized_value.py_inline())
        return PyEvalCode("{" + ", ".join(serialized_items) + "}")

    def _serialize_dim(self, dim: Dim, prefix: str) -> Union[PyEvalCode, PyCode]:
        # we serialize a Dim object, so we know that RETURNN is available for import
        from returnn.tensor import Dim, batch_dim, single_step_dim

        assert isinstance(dim, Dim)

        # See also returnn_common.nn.naming.ReturnnDimTagsProxy.dim_ref_repr
        # and returnn_common.nn.naming.ReturnnDimTagsProxy.DimRefProxy.dim_repr.
        if dim == batch_dim:
            return self._serialize_global(dim, prefix, mod_name="returnn.tensor", qualname="batch_dim")
        if dim == single_step_dim:
            return self._serialize_global(dim, prefix, mod_name="returnn.tensor", qualname="single_step_dim")

        if dim.match_priority:
            base_dim_str = self._serialize_value(dim.copy(match_priority=0), prefix=f"{prefix}_p0", recursive=True)
            assert isinstance(base_dim_str, PyEvalCode)
            return PyEvalCode(f"{base_dim_str.py_inline()}.copy(match_priority={dim.match_priority})")
        if not dim.derived_from_op and dim.get_same_base().derived_from_op:
            dim = dim.get_same_base()

        if dim.derived_from_op:
            if dim.derived_from_op.kind == "constant":
                v = dim.derived_from_op.attribs["value"]
                return PyEvalCode(str(v), need_brackets_when_inlined=v < 0)
            func_map = {"truediv_left": "div_left", "ceildiv_left": "ceildiv_left", "ceildiv_right": "ceildiv_right"}
            inputs_s: List[PyEvalCode] = [
                self._serialize_value(x, prefix=f"{prefix}_in{i}", recursive=True)
                for i, x in enumerate(dim.derived_from_op.inputs)
            ]
            assert all(isinstance(x, PyEvalCode) for x in inputs_s)
            if dim.derived_from_op.kind in func_map:
                assert len(dim.derived_from_op.inputs) == 2
                a, b = inputs_s
                a: PyEvalCode
                b: PyEvalCode
                return PyEvalCode(f"{a.py_inline()}.{func_map[dim.derived_from_op.kind]}({b.py_inline()})")
            op_str = {"add": "+", "mul": "*", "truediv_right": "//", "floordiv_right": "//"}[dim.derived_from_op.kind]
            s = f" {op_str} ".join(x.py_inline() for x in inputs_s)
            return PyEvalCode(s, need_brackets_when_inlined=True)

        # generic fallback
        dim_type_str = self._serialize_value(type(dim), prefix="Dim", recursive=True)
        assert isinstance(dim_type_str, PyEvalCode)
        kwargs = {"name": repr(dim.name)}
        if dim.kind is not None:
            kind_s = {Dim.Types.Batch: "Batch", Dim.Types.Spatial: "Spatial", Dim.Types.Feature: "Feature"}[dim.kind]
            kwargs["kind"] = f"{dim_type_str.py_inline()}.Types.{kind_s}"
        return PyEvalCode(
            f"{dim_type_str.py_inline()}"
            f"({dim.dimension}, {', '.join(f'{key}={value}' for key, value in kwargs.items())})"
        )

    def _serialize_cached_file(self, cached_file: CachedFile) -> PyEvalCode:
        # we serialize a CachedFile object, so we know that RETURNN is available for import
        from returnn.util.file_cache import CachedFile

        assert isinstance(cached_file, CachedFile)
        cf_type_str = self._serialize_value(type(cached_file), prefix="CachedFile", recursive=True)
        assert isinstance(cf_type_str, PyEvalCode)
        assert isinstance(cached_file.filename, str)
        return PyEvalCode(f"{cf_type_str.py_inline()}({repr(cached_file.filename)})")

    def _serialize_global(
        self, value: Any, name: str, *, mod_name: Optional[str] = None, qualname: Optional[str] = None
    ) -> Union[PyEvalCode, PyCode]:
        mod_name = mod_name or getattr(value, "__module__", None)
        if not mod_name:
            raise SerializationError(f"cannot handle {value!r} (type {type(value).__name__}) as global, no __module__")
        mod = sys.modules.get(mod_name)
        if not mod:
            raise SerializationError(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, unknown __module__ {mod_name!r}"
            )
        qualname = qualname or getattr(value, "__qualname__", None)
        if not qualname:
            raise SerializationError(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, no __qualname__"
            )
        qualname_parts = qualname.split(".")
        obj = [mod]
        for i in range(len(qualname_parts)):
            if not hasattr(obj[-1], qualname_parts[i]):
                raise SerializationError(
                    f"cannot handle {value!r} (type {type(value).__name__}) as global,"
                    f" qualname {qualname} not found,"
                    f" no {'.'.join(qualname_parts[: i + 1])} in module {mod_name}"
                )
            obj.append(getattr(obj[-1], qualname_parts[i]))
        if obj[-1] is not value:
            raise SerializationError(
                f"cannot handle {value!r} (type {type(value).__name__}) as global,"
                f" qualname {qualname} gives different object {obj[-1]!r}"
            )
        if len(qualname_parts) > 1:
            base_obj_repr = self._serialize_value(obj[-2], prefix=name + "_base")
            return PyEvalCode(f"{base_obj_repr.py_inline()}.{qualname_parts[-1]}")
        if "." in mod_name:
            # Maybe we can shorten the import.
            # Check if some of the parent modules already import the object.
            mod_name_parts = mod_name.split(".")
            for i in range(len(mod_name_parts)):
                parent_mod_name = ".".join(mod_name_parts[: i + 1])
                mod = sys.modules.get(parent_mod_name)
                if mod and getattr(mod, qualname, None) is value:
                    mod_name = parent_mod_name  # we can directly use this
                    break
        self._setup_module_import(mod_name)
        return PyCode(
            py_name=name,
            value=value,
            py_code=f"from {mod_name} import {qualname}\n"
            if qualname == name
            else f"from {mod_name} import {qualname} as {name}\n",
        )

    def _serialize_module(self, value: ModuleType, name: str) -> PyCode:
        mod_name = value.__name__
        assert sys.modules[mod_name] is value
        self._setup_module_import(mod_name)
        if "." not in mod_name:
            return PyCode(
                py_name=name,
                value=value,
                py_code=f"import {mod_name}\n" if mod_name == name else f"import {mod_name} as {name}\n",
            )
        mod_name, qualname = mod_name.rsplit(".", 1)
        return PyCode(
            py_name=name,
            value=value,
            py_code=f"from {mod_name} import {qualname}\n"
            if qualname == name
            else f"from {mod_name} import {qualname} as {name}\n",
        )

    def _setup_module_import(self, mod_name: str):
        """make sure that the import works, by preparing ``sys.path`` if necessary"""
        if "." in mod_name:
            mod_name = mod_name.split(".", 1)[0]
        if mod_name in self.known_modules:
            return
        mod = sys.modules[mod_name]
        if not hasattr(mod, "__file__"):
            return  # assume builtin module or so
        mod_path = _get_module_path_from_module(mod)
        self.add_sys_path(mod_path)
        self.known_modules.add(mod_name)

    def add_sys_path(self, path: str, *, recursive: bool = True):
        """
        Add an entry to sys.path if it is not already there.
        """
        if path in self.added_sys_paths:
            return  # already added
        base_sys_path = [path_ for path_ in _get_base_sys_path_list() if path_]
        assert base_sys_path
        if path in base_sys_path:
            return  # already in (base) sys.path

        if not recursive:
            self._handle_next_queue_item(_AssignQueueItem(sys))
        sys_s = self._serialize_value(sys, prefix="sys")
        assert isinstance(sys_s, PyEvalCode)
        if path in sys.path:
            path_index = sys.path.index(path)
            assert base_sys_path[0] in sys.path, f"sys.path {sys.path} does not contain {base_sys_path[0]!r}"
            base_sys_path_index = sys.path.index(base_sys_path[0])
            if path_index < base_sys_path_index:
                insert_idx = self._next_sys_path_insert_idx
                self._next_sys_path_insert_idx += 1
            else:
                insert_idx = None  # add at the end
        else:
            # Maybe some other import mechanism is in place (like the Sisyphus config loading mechanism),
            # which takes precedence over sys.path.
            # Thus put it in front of the base sys.path.
            insert_idx = self._next_sys_path_insert_idx
            self._next_sys_path_insert_idx += 1
        if insert_idx is not None:
            code = PyCode(
                py_name=None, value=None, py_code=f"{sys_s.py_inline()}.path.insert({insert_idx}, {path!r})\n"
            )
        else:
            code = PyCode(py_name=None, value=None, py_code=f"{sys_s.py_inline()}.path.append({path!r})\n")
        code.idx = self._next_assignment_idx
        self._next_assignment_idx += 1
        self.assignments_dict_by_idx[code.idx] = code
        self.added_sys_paths.add(path)

    def _serialize_functools_partial(self, value: functools.partial, name: str) -> PyEvalCode:
        # The generic fallback using __reduce__ would also work with this.
        # However, the following is a bit nicer in the generated code.
        mod_s = self._serialize_value(functools, prefix="functools")
        assert isinstance(mod_s, PyEvalCode)
        func_s = self._serialize_value(value.func, prefix=f"{name}_func", recursive=True)
        assert isinstance(func_s, PyEvalCode)
        args_s = [
            self._serialize_value(arg, prefix=f"{name}_arg{i}", recursive=True) for i, arg in enumerate(value.args)
        ]
        dictitems_s = []
        for key, value_ in value.keywords.items():
            assert isinstance(key, str) and _is_valid_python_identifier_name(key)
            serialized_value = self._serialize_value(value_, prefix=f"{name}_{key}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            dictitems_s.append((key, serialized_value))
        args_ss = "".join(f", {arg_s.py_inline()}" for arg_s in args_s)
        dictitems_ss = "".join(f", {k}={v.py_inline()}" for k, v in dictitems_s)
        return PyEvalCode(f"{mod_s.py_inline()}.partial({func_s.py_inline()}{args_ss}{dictitems_ss})")

    def _serialize_method(self, value: MethodType, name: str) -> PyEvalCode:
        mod_s = self._serialize_value(types, prefix="types")
        assert isinstance(mod_s, PyEvalCode)
        func_s = self._serialize_value(value.__func__, prefix=f"{name}_func", recursive=True)
        assert isinstance(func_s, PyEvalCode)
        self_s = self._serialize_value(value.__self__, prefix=f"{name}_self", recursive=True)
        assert isinstance(self_s, PyEvalCode)
        return PyEvalCode(f"{mod_s.py_inline()}.MethodType({func_s.py_inline()}, {self_s.py_inline()})")

    def _serialize_call(self, value: Call, name: str) -> PyEvalCode:
        assert isinstance(value, Call)
        unhashed_args_s = self._serialize_value(
            dict(value.unhashed_kwargs), recursive=True, prefix=f"{name}_unhashed_kwargs"
        )
        assert isinstance(unhashed_args_s, PyEvalCode)
        hashed_args_s = self._serialize_value(dict(value.kwargs), recursive=True, prefix=f"{name}_hashed_kwargs")
        assert isinstance(hashed_args_s, PyEvalCode)
        call_str = f"{value.callable_name}(**{unhashed_args_s.py_inline()}, **{hashed_args_s.py_inline()})"
        if value.return_assign_variables is not None:
            assert isinstance(value.return_assign_variables, str) or len(value.return_assign_variables) == 1, (
                "cannot destructure-assign when auto serialization is used"
            )
            return PyCode(name, value, py_code=f"{name} = {call_str}\n", py_value_repr=PyEvalCode(call_str))
        return PyEvalCode(call_str)

    def _serialize_call_import(self, value: CallImport, name: str) -> PyCode:
        assert isinstance(value, CallImport)
        func_s = PyEvalCode(f'__import__("{value.module}", fromlist=["{value.object_name}"]).{value.object_name}')

        dictitems_s = []
        for key, value_ in {**value.unhashed_arguments, **value.hashed_arguments}.items():
            assert isinstance(key, str) and _is_valid_python_identifier_name(key)
            serialized_value = self._serialize_value(value_, prefix=f"{name}_{key}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            dictitems_s.append((key, serialized_value))
        dictitems_ss = ", ".join(f"{k}={v.py_inline()}" for k, v in dictitems_s)

        repr_code = f"{func_s.py_inline()}({dictitems_ss})"
        return PyCode(
            name,
            value,
            py_code=f"{name} = {repr_code}\n",
            py_value_repr=PyEvalCode(repr_code),
        )

    def _serialize_partial_import(self, value: PartialImport, name: str) -> PyCode:
        assert isinstance(value, PartialImport)
        mod_s = self._serialize_value(functools, prefix="functools")
        assert isinstance(mod_s, PyEvalCode)
        func_s = PyEvalCode(f'__import__("{value.module}", fromlist=["{value.object_name}"]).{value.object_name}')

        dictitems_s = []
        for key, value_ in {**value.unhashed_arguments, **value.hashed_arguments}.items():
            assert isinstance(key, str) and _is_valid_python_identifier_name(key)
            serialized_value = self._serialize_value(value_, prefix=f"{name}_{key}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            dictitems_s.append((key, serialized_value))
        dictitems_ss = "".join(f", {k}={v.py_inline()}" for k, v in dictitems_s)

        repr_code = f"{mod_s.py_inline()}.partial({func_s.py_inline()}, {dictitems_ss})"
        return PyCode(
            name,
            value,
            py_code=f"{name} = ${repr_code}\n",
            py_value_repr=PyEvalCode(repr_code),
        )

    def _serialize_import(self, value: Import, name: str) -> PyCode:
        assert isinstance(value, Import)
        return PyCode(
            py_name=name,
            value=value,
            py_code=f"from {value.module} import {value.object_name}\n"
            if value.object_name == name
            else f"from {value.module} import {value.object_name} as {name}\n",
        )

    def _serialize_code_from_function(self, value: CodeFromFunction, name: str) -> PyCode:
        assert isinstance(value, CodeFromFunction)
        func_with_legal_name = CodeFromFunction(name=name, func=value.func)
        return PyCode(py_name=name, value=value, py_code=func_with_legal_name.get())

    def _serialize_code_wrapper(self, value: CodeWrapper) -> PyEvalCode:
        assert isinstance(value, CodeWrapper)
        code = value.code
        if isinstance(code, DelayedBase):
            code = code.get()
        return PyEvalCode(code, need_brackets_when_inlined=False)

    def _serialize_sis_path(self, value: Path) -> PyEvalCode:
        assert isinstance(value, Path)
        assert self.sis_path_handling  # should not call this otherwise
        if self.sis_path_handling == SisPathHandling.NO_DEPS:
            path_type_str = self._serialize_value(type(value), prefix="Path", recursive=True)
            assert isinstance(path_type_str, PyEvalCode)
            return PyEvalCode(f"{path_type_str.py_inline()}({value.get_path()!r})")
        elif self.sis_path_handling == SisPathHandling.AS_STRING:
            # Note: If we would want to have Sisyphus file_caching support here,
            # we could also refer to that file_caching function,
            # and call it here in the generated code.
            return PyEvalCode(repr(value.get_path()))
        else:
            raise ValueError(f"invalid sis_path_handling {self.sis_path_handling}")

    def _serialize_checkpoint(self, value: Union[PtCheckpoint, Checkpoint]) -> PyEvalCode:
        assert isinstance(value, (Checkpoint, PtCheckpoint))
        return PyEvalCode(repr(value), need_brackets_when_inlined=False)


class _SerializationDependsOnNotYetSerializedOtherVarException(Exception):
    def __init__(self, queue_item: _AssignQueueItem):
        super().__init__(
            f"serialization depends on not yet serialized other var:"
            f" ({queue_item.suggested_var_name}) = {queue_item.value!r} (type {type(queue_item.value).__name__})"
        )
        self.queue_item = queue_item


class SerializationError(Exception):
    """
    Cannot serialize this object.
    """


@dataclass
class _AssignQueueItem:
    value: Any
    required_var_name: Optional[str] = None
    suggested_var_name: Optional[str] = None

    via_queue_item: Optional[_AssignQueueItem] = None  # for debugging

    def debug_trace(self) -> str:
        if self.via_queue_item:
            return (
                f"{self.required_var_name or self.suggested_var_name} <{type(self.value).__name__}>"
                f" -> {self.via_queue_item.debug_trace()}"
            )
        return f"config {self.required_var_name or self.suggested_var_name} = {self.value!r}"


@dataclass
class PyCode(SerializerObject):
    """
    The Python code will always assign some variable.

    E.g.::

        x = 42  # assign `x`

    Or::

        def f(): ...  # assign `f`

    Or::

        import sys  # assign `sys`
    """

    py_name: Optional[str]
    value: Any
    py_code: str
    py_value_repr: Optional[PyEvalCode] = None
    is_direct_config_entry: bool = False
    use_for_hash: bool = False
    ref_count: int = 0  # by other statements
    idx: Optional[int] = None
    has_later_state_setup: bool = False

    def get(self) -> str:
        return self.py_code

    def _sis_hash(self) -> bytes:
        if not self.use_for_hash:
            raise Exception(f"{self} should not be hashed. Maybe wrap this in a serialization Collection")
        return sis_hash_helper((self.py_name, self.value))


@dataclass
class PyEvalCode:
    """
    When some repr can represent the value directly.
    """

    py_value_repr: str
    need_brackets_when_inlined: bool = False  # e.g. for math expressions like `a + b`

    def py_inline(self) -> str:
        return f"({self.py_value_repr})" if self.need_brackets_when_inlined else self.py_value_repr


@dataclass
class _PyCodeWithDeferredStateQueueItem:
    code: PyCode
    extra: _DeferredStateQueueItem


@dataclass
class _DeferredStateQueueItem:
    py_name: str
    value: Any
    # These are the extra args from the __reduce__ or __reduce_ex__.
    state: Any
    listitems: Optional[Sequence[Any]] = None
    dictitems: Optional[Sequence[Tuple[Any, Any]]] = None
    state_setter: Optional[Any] = None

    via_queue_item: Optional[_AssignQueueItem] = None  # only for debugging

    def debug_trace(self) -> str:
        if self.via_queue_item:
            return f"{self.py_name} <{type(self.value).__name__}> -> {self.via_queue_item.debug_trace()}"
        return "<unknown>"


class _Ref:
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"_Ref({self.value!r})"

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other: _Ref):
        return self.value is other.value

    def __ne__(self, other: _Ref):
        return not (self == other)


_base_sys_path_list: Optional[str] = None


def _get_base_sys_path_list() -> List[str]:
    global _base_sys_path_list
    if _base_sys_path_list is None:
        env_copy = os.environ.copy()
        env_copy.pop("PYTHONPATH", None)
        _base_sys_path_list = eval(
            subprocess.check_output([sys.executable, "-c", "import sys; print(sys.path)"], env=env_copy)
            .decode("utf8")
            .strip()
        )
        assert isinstance(_base_sys_path_list, list) and all(isinstance(p, str) for p in _base_sys_path_list)
    return _base_sys_path_list


def _get_module_path_from_module(mod: ModuleType) -> str:
    mod_filename = mod.__file__
    if mod_filename.endswith("/__init__.py"):
        mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])
    else:
        mod_path = os.path.dirname(mod_filename)
    return mod_path


def _isinstance_returnn_obj(obj: Any) -> bool:
    mod_s = getattr(type(obj), "__module__", None)
    return mod_s is not None and mod_s.startswith("returnn.")


def _isinstance_returnn_cached_file(obj: Any) -> bool:
    if not _isinstance_returnn_obj(obj):
        return False

    from returnn.util.file_cache import CachedFile

    return isinstance(obj, CachedFile)


def _isinstance_returnn_dim(obj: Any) -> bool:
    if not _isinstance_returnn_obj(obj):
        return False

    from returnn.tensor import Dim

    return isinstance(obj, Dim)
