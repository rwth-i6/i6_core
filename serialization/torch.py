"""
Helpers that are specific for i6_models/PyTorch setups
"""

from __future__ import annotations

__all__ = ["build_i6_models_config_constructor_serializers"]

from collections import OrderedDict
from dataclasses import fields
from inspect import isfunction
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from sisyphus.delayed_ops import DelayedBase

if TYPE_CHECKING:
    from i6_models.config import ModelConfiguration

from .base import Call, Import


def build_i6_models_config_constructor_serializers(
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
