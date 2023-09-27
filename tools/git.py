__all__ = ["CloneGitRepositoryJob"]

import logging
import subprocess as sp
from typing import Optional

from sisyphus import *

Path = setup_path(__package__)


class CloneGitRepositoryJob(Job):
    """
    Clone a git repository given optional branch name and commit hash
    """

    __sis_hash_exclude__ = {"clone_submodules": False}

    def __init__(
        self,
        url: str,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        checkout_folder_name: str = "repository",
        clone_submodules: bool = False,
    ):
        """

        :param url: Git repository url
        :param branch: Git branch name
        :param commit: Git commit hash
        :param checkout_folder_name: Name of the output path repository folder
        :param clone_submodules: Flag to clone submodules if set to True
        """
        self.url = url
        self.branch = branch
        self.commit = commit
        self.clone_submodules = clone_submodules

        self.out_repository = self.output_path(checkout_folder_name, True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        args = ["git", "clone", self.url]
        if self.branch is not None:
            args.extend(["-b", self.branch])
        repository_dir = self.out_repository.get_path()
        args += [repository_dir]
        logging.info("running command: %s" % " ".join(args))
        sp.run(args, check=True)

        if self.commit is not None:
            args = ["git", "checkout", self.commit]
            logging.info("running command: %s" % " ".join(args))
            sp.run(args, cwd=repository_dir, check=True)

        if self.clone_submodules:
            args = ["git", "submodule", "update", "--init", "--recursive"]
            sp.run(args, cwd=repository_dir, check=True)
