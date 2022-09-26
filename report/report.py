from sisyphus import *
import sys
import os
import getpass
import pprint
import gzip
from typing import Dict, Union, Callable
import time
import json

_Report_Type = Dict[str, Union[tk.Path, any]]


class ReportResultsJob(Job):
    """
    Job to report results either via mail and/or just condensed in a file
    """

    def __init__(
        self,
        name: str,
        report: _Report_Type,
        report_dir: tk.Path = None,
        report_format: Callable[[_Report_Type], str] = None,
        recipe_path=Path("."),
    ):
        """

        :param name:
        :param dict report: dictionary containing the report files
        :param report_dir:
        :param (report: dict) -> List[str] report_format:
        :param recipe_path:
        """
        self.name = name
        self.report = report
        self.report_dir = report_dir
        self.report_format = report_format
        self.recipe_path = recipe_path

        self.out_report = self.output_path("report.gz")

        self.sis_command_line = sys.argv
        self.cwd = os.path.abspath(".")
        self.config_path = tk.config_manager.current_config
        assert self.config_path is not None, "Could not find config path"
        try:
            with open(self.config_path) as f:
                self.config = f.read()
        except IOError as e:
            if e.errno != 2:
                raise e
            else:
                self.config = self.config_path

        with open(gs.GLOBAL_SETTINGS_FILE) as f:
            self.settings = f.read()

        for i in [
            "recipe_hash",
            "recipe_date",
            "date",
            "user",
            "name",
            "config",
            "config_path",
            "config_line",
            "settings",
            "cwd",
            "sis_command_line",
        ]:
            assert i not in report, "%s will be set automatically"

    def run(self):
        user = getpass.getuser()
        report = self.report.copy()

        report["date"] = time.time()
        report["user"] = user
        report["name"] = self.name
        report["config_path"] = self.config_path
        report["config"] = self.config
        report["settings"] = self.settings
        report["sis_command_line"] = self.sis_command_line
        report["cwd"] = self.cwd

        if self.report_format is None:
            self.report_format = pprint.pformat

        with gzip.open(str(self.out_report), "w") as f:
            f.write(self.report_format(report).encode() + b"\n")

        if self.report_dir:
            report_file_name = "%s.%s.%s.gz" % (
                user,
                self.config_path.replace("/", "_"),
                self.name,
            )
            report_path = os.path.join(self.report_dir, report_file_name)
            print("Write report to %s" % report_path)
            with gzip.open(report_path, "w") as f:
                f.write(json.dumps(report).encode() + b"\n")

        self.mail_address = getattr(gs, "MAIL_ADDRESS", None)
        if self.mail_address:
            self.sh(
                "zcat {out_report} | mail -s 'Report finished: {name} {config_path}' {mail_address}"
            )

    def tasks(self):
        yield Task("run", mini_task=True)


def gmm_example_report_format(report: Dict[str, tk.Variable]) -> str:
    """
    Example report format for a GMM evaluated on dev-clean and dev-other
    :param report:
    :return:
    """
    results = {
        "dev-clean": {
            "Monophone": {},
            "Triphone": {},
            "SAT": {},
            "VTLN": {},
            "VTLN+SAT": {},
        },
        "dev-other": {
            "Monophone": {},
            "Triphone": {},
            "SAT": {},
            "VTLN": {},
            "VTLN+SAT": {},
        },
    }
    for step_name, score in report.items():
        if "dev-clean" in step_name:
            set = "dev-clean"
        else:
            set = "dev-other"
        if not step_name.startswith("scorer"):
            continue
        if "mono" in step_name:
            step = "Monophone"
        elif "tri" in step_name:
            step = "Triphone"
        elif "vtln+sat" in step_name:
            step = "VTLN+SAT"
        elif "sat" in step_name:
            step = "SAT"
        else:
            step = "VTLN"
        if "iter08" in step_name:
            results[set][step]["08"] = score
        elif "iter10" in step_name:
            results[set][step]["10"] = score

    out = []
    out.append(
        f"""Name: {report["name"]}
          Path: {report["config_path"]}
          Date: {report["date"]}

          Results:"""
    )
    out.append("Step".ljust(20) + "dev-clean".ljust(10) + "dev-other")
    out.append(
        "Monophone".ljust(16)
        + str(results["dev-clean"]["Monophone"]["10"]).ljust(14)
        + str(results["dev-other"]["Monophone"]["10"])
    )
    out.append(
        "Triphone 08".ljust(19)
        + str(results["dev-clean"]["Triphone"]["08"]).ljust(14)
        + str(results["dev-other"]["Triphone"]["08"])
    )
    out.append(
        "Triphone 10".ljust(19)
        + str(results["dev-clean"]["Triphone"]["10"]).ljust(14)
        + str(results["dev-other"]["Triphone"]["10"])
    )
    out.append(
        "VTLN 08".ljust(21)
        + str(results["dev-clean"]["VTLN"]["08"]).ljust(14)
        + str(results["dev-other"]["VTLN"]["08"])
    )
    out.append(
        "VTLN 10".ljust(21)
        + str(results["dev-clean"]["VTLN"]["10"]).ljust(14)
        + str(results["dev-other"]["VTLN"]["10"])
    )
    out.append(
        "SAT 08".ljust(23)
        + str(results["dev-clean"]["SAT"]["08"]).ljust(14)
        + str(results["dev-other"]["SAT"]["08"])
    )
    out.append(
        "SAT 10".ljust(23)
        + str(results["dev-clean"]["SAT"]["10"]).ljust(14)
        + str(results["dev-other"]["SAT"]["10"])
    )
    out.append(
        "VTLN+SAT 08".ljust(17)
        + str(results["dev-clean"]["VTLN+SAT"]["08"]).ljust(14)
        + str(results["dev-other"]["VTLN+SAT"]["08"])
    )
    out.append(
        "VTLN+SAT 10".ljust(17)
        + str(results["dev-clean"]["VTLN+SAT"]["10"]).ljust(14)
        + str(results["dev-other"]["VTLN+SAT"]["10"])
    )

    return "\n".join(out)
