__all__ = ["CloneGitRepositoryJob"]

import logging
import subprocess as sp
from typing import Optional
import os

from sisyphus import *

Path = setup_path(__package__)


class CloneGitRepositoryJob(Job):
    """
    Clone a git repository given optional branch name and commit hash
    """

    __sis_hash_exclude__ = {"clone_submodules": False, "files_to_checkout": None}

    def __init__(
        self,
        url: str,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        checkout_folder_name: str = "repository",
        clone_submodules: bool = False,
        files_to_checkout: Optional[list[str]] = None,
    ):
        """

        :param url: Git repository url
        :param branch: Git branch name
        :param commit: Git commit hash
        :param checkout_folder_name: Name of the output path repository folder
        :param clone_submodules: Flag to clone submodules if set to True
        :param files_to_checkout: List of files to be checked out sparsely. If not set, the entire repo is checked out (default behaviour).
        """
        self.url = url
        self.branch = branch
        self.commit = commit
        self.clone_submodules = clone_submodules
        self.files_to_checkout = files_to_checkout

        self.out_repository = self.output_path(checkout_folder_name, True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.files_to_checkout is not None:
            args = ["git", "clone", "--no-checkout", self.url]
        else:
            args = ["git", "clone", self.url]
        if self.branch is not None:
            args.extend(["-b", self.branch])
        repository_dir = self.out_repository.get_path()
        args += [repository_dir]
        logging.info("running command: %s" % " ".join(args))
        sp.run(args, check=True)

        if self.files_to_checkout is not None:
            args = ["git", "checkout"]
            commit = self.commit if self.commit is not None else ""
            args.extend([commit, "--"] + self.files_to_checkout)
            logging.info("running command: %s" % " ".join(args))
            sp.run(args, cwd=repository_dir, check=True)
            for file in self.files_to_checkout:
                # some files may be links, so download the original file to avoid a missing symlink target
                if os.path.islink(f"{repository_dir}/{file}"):
                    args = ["git", "checkout", commit, "--", os.path.realpath(f"{repository_dir}/{file}")]
                    logging.info("running command: %s" % " ".join(args))
                    sp.run(args, cwd=repository_dir, check=True)
        elif self.commit is not None:
            args = ["git", "checkout", self.commit]
            logging.info("running command: %s" % " ".join(args))
            sp.run(args, cwd=repository_dir, check=True)

        if self.clone_submodules:
            args = ["git", "submodule", "update", "--init", "--recursive"]
            sp.run(args, cwd=repository_dir, check=True)
