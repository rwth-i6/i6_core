__all__ = ["get_returnn_root"]

import logging
from typing import Optional

from sisyphus import gs, Path

import i6_core.tools as tools

RETURNN_COMMIT = None


def get_returnn_root(returnn_root: Optional[Path]) -> Path:
    """
    Provides a tk.Path object that points to the root of a returnn repository.
    Will return the first location it finds:
        1. the argument provided to get_returnn_root
        2. gs.RETURNN_ROOT
        3. checkout current head

    :param returnn_root: Path to the root of a returnn repository or None
    :returns: Path to the root of a returnn repository
    """
    if isinstance(returnn_root, Path):
        return returnn_root
    elif isinstance(returnn_root, str):
        logging.warning("Giving returnn_root as str is deprecated, use tk.Path instead")
        return Path(returnn_root)
    elif returnn_root is None:
        if hasattr(gs, "RETURNN_ROOT"):
            logging.warning(
                "Using gs.RETURNN_ROOT is deprecated, please provide an explicit tk.Path object"
            )
            return Path(gs.RETURNN_ROOT)
        else:
            returnn_root = tools.CloneGitRepositoryJob(
                url="https://github.com/rwth-i6/returnn.git",
                commit=RETURNN_COMMIT,
                checkout_folder_name="returnn",
            ).out_repository
            return returnn_root
    else:
        assert False, "unsupported type given for returnn_root, use tk.Path"
