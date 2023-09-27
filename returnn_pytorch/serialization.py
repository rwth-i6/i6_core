from __future__ import annotations

import os
import pathlib
import shutil
import string
import textwrap
from collections import OrderedDict
from dataclasses import fields
from inspect import isfunction
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import torch
from i6_core.util import instanciate_delayed
from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

if TYPE_CHECKING:
    from i6_models.config import ModelConfiguration

from ..serialization import Call, Import, SerializerObject


class PyTorchModel(SerializerObject):
    """
    Serializes a `get_network` function into the config, which calls
    a defined network construction function and defines the parameters to it.
    This is for returnn_common networks.

    Note that the network constructor function always needs "epoch" as first defined parameter,
    and should return an `nn.Module` object.
    """

    TEMPLATE = textwrap.dedent(
        """\

    model_kwargs = ${MODEL_KWARGS}

    def get_model(epoch, step, **kwargs):
        return ${MODEL_CLASS}(epoch=epoch, step=step, **model_kwargs, **kwargs)

    """
    )

    def __init__(
        self,
        model_class_name: str,
        model_kwargs: Dict[str, Any],
    ):
        """
        :param model_class_name:
        :param model_kwargs:
        """

        super().__init__()
        self.model_class_name = model_class_name
        self.model_kwargs = model_kwargs

    def get(self):
        """get"""
        return string.Template(self.TEMPLATE).substitute(
            {
                "MODEL_KWARGS": str(instanciate_delayed(self.model_kwargs)),
                "MODEL_CLASS": self.model_class_name,
            }
        )

    def _sis_hash(self):
        h = {
            "model_kwargs": self.model_kwargs,
        }
        return sis_hash_helper(h)


class Collection(DelayedBase):
    """
    A helper class to serialize a RETURNN config with returnn_common elements.
    Should be passed to either `returnn_prolog` or `returnn_epilog` in a `ReturnnConfig`

    The tasks are:
     - managing the returnn_common version (via sys.path)
     - managing returnn_common net definitions (via importing a nn.Module class definition, using :class:`Import`)
     - managing returnn_common net construction which returns the final net, using `ReturnnCommonImport`
       (via importing a (epoch, **kwargs) -> nn.Module function)
     - managing nn.Dim/Data and the extern_data entry
       via :class:`ExternData`, :class:`DimInitArgs` and :class:`DataInitArgs`
     - managing the package imports from which all imports can be found
     - optionally make a local copy of all imported code instead if importing it directly out of the recipes

    """

    def __init__(
        self,
        serializer_objects: List[SerializerObject],
        *,
        packages: Optional[Set[Union[str, tk.Path]]] = None,
        make_local_package_copy: bool = False,
    ):
        """

        :param serializer_objects: all serializer objects which are serialized into a string in order
        :param packages: Path to packages to import, if None, tries to extract them from serializer_objects
        :param make_local_package_copy: whether to make a local copy of imported code into the Job directory
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects
        self.packages = packages
        self.make_local_package_copy = make_local_package_copy

        self.root_path = os.path.join(gs.BASE_DIR, gs.RECIPE_PREFIX)

        assert (not self.make_local_package_copy) or self.packages, (
            "Please specify which packages to copy if you are using "
            "`make_local_package_copy=True` in combination with `Import` objects"
        )

    def get(self) -> str:
        """get"""
        content = ["import os\nimport sys\n"]

        # have sys.path setup first
        if self.make_local_package_copy and self.packages is not None:
            out_dir = os.path.join(os.getcwd(), "../output")
            for package in self.packages:
                if isinstance(package, tk.Path):
                    package_path = package.get_path()
                elif isinstance(package, str):
                    package_path = package.replace(".", "/")
                else:
                    assert False, "invalid type for packages"
                target_package_path = os.path.join(out_dir, package_path)
                pathlib.Path(os.path.dirname(target_package_path)).mkdir(parents=True, exist_ok=True)
                shutil.copytree(os.path.join(self.root_path, package_path), target_package_path)
                content.append(f"sys.path.insert(0, os.path.dirname(__file__))\n")
        else:
            content.append(f"sys.path.insert(0, {self.root_path!r})\n")

        content += [obj.get() for obj in self.serializer_objects]
        return "".join(content)

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [obj for obj in self.serializer_objects if obj.use_for_hash],
        }
        return sis_hash_helper(h)


def build_config_constructor_serializers(
    cfg: ModelConfiguration, variable_name: Optional[str] = None, unhashed_package_root: Optional[str] = None
) -> Tuple[Call, List[Import]]:
    """
    Creates a Call object that will re-construct the given ModelConfiguration when serialized and
    optionally assigns the resulting config object to a variable. Automatically generates a list of all
    necessary imports in order to perform the constructor call.

    :param cfg: ModelConfiguration object that will be re-constructed by the Call serializer
    :param variable_name: Name of the variable which the constructed ModelConfiguration
                          will be assigned to. If None, the result will not be assigned
                          to a variable.
    :param unhashed_package_root: Will be passed to all generated Import objects.
    :return: Call object and list of necessary imports.
    """
    from i6_models.config import ModelConfiguration, ModuleFactoryV1

    # Import the class of <cfg>
    imports = [
        Import(
            code_object_path=f"{type(cfg).__module__}.{type(cfg).__name__}", unhashed_package_root=unhashed_package_root
        )
    ]

    call_kwargs = []

    # Iterate over all dataclass fields
    for key in fields(type(cfg)):
        # Value corresponding to dataclass field name
        value = getattr(cfg, key.name)

        # Switch over serialization logic for different subtypes
        if isinstance(value, ModelConfiguration):
            # Example:
            # ConformerBlockConfig(mhsa_config=ConformerMHSAConfig(...))
            # -> Sub-Constructor-Call and imports for ConformerMHSAConfig
            subcall, subimports = build_config_constructor_serializers(value)
            imports += subimports
            call_kwargs.append((key.name, subcall))
        elif isinstance(value, ModuleFactoryV1):
            # Example:
            # ConformerEncoderConfig(
            #     frontend=ModuleFactoryV1(module_class=VGGFrontend, cfg=VGGFrontendConfig(...)))
            # -> Import classes ModuleFactoryV1, VGGFrontend and VGGFrontendConfig
            # -> Sub-Constructor-Call for VGGFrontendConfig
            subcall, subimports = build_config_constructor_serializers(value.cfg)
            imports += subimports
            imports.append(
                Import(
                    code_object_path=f"{value.module_class.__module__}.{value.module_class.__name__}",
                    unhashed_package_root=unhashed_package_root,
                )
            )
            imports.append(
                Import(
                    code_object_path=f"{ModuleFactoryV1.__module__}.{ModuleFactoryV1.__name__}",
                    unhashed_package_root=unhashed_package_root,
                )
            )
            call_kwargs.append(
                (
                    key.name,
                    Call(
                        callable_name=ModuleFactoryV1.__name__,
                        kwargs=[("module_class", value.module_class.__name__), ("cfg", subcall)],
                    ),
                )
            )
        elif isinstance(value, torch.nn.Module):
            # Example:
            # ConformerConvolutionConfig(norm=BatchNorm1d(...))
            # -> Import class BatchNorm1d
            # -> Sub-serialization of BatchNorm1d object.
            #       The __str__ function of torch.nn.Module already does this in the way we want.
            imports.append(
                Import(
                    code_object_path=f"{value.__module__}.{type(value).__name__}",
                    unhashed_package_root=unhashed_package_root,
                )
            )
            call_kwargs.append((key.name, str(value)))
        elif isfunction(value):
            # Example:
            # ConformerConvolutionConfig(activation=torch.nn.functional.silu)
            # -> Import function silu
            # Builtins (e.g. 'sum') do not need to be imported
            if value.__module__ != "builtins":
                imports.append(
                    Import(
                        code_object_path=f"{value.__module__}.{value.__name__}",
                        unhashed_package_root=unhashed_package_root,
                    )
                )
            call_kwargs.append((key.name, value.__name__))
        elif isinstance(value, DelayedBase):
            # sisyphus variables are just given as-is and will be instanciated only when calling "get".
            call_kwargs.append((key.name, value))
        else:
            # No special case (usually python primitives)
            # -> Just get string representation
            call_kwargs.append((key.name, str(value)))

    imports = list(OrderedDict.fromkeys(imports))  # remove duplications

    return Call(callable_name=type(cfg).__name__, kwargs=call_kwargs, return_assign_variables=variable_name), imports
