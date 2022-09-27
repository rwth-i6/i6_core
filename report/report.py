from sisyphus import *

import sys
import getpass
import gzip
from datetime import datetime
from typing import Dict, Union, Callable, Optional, Iterable

_Report_Type = Dict[str, Union[tk.AbstractPath, str]]


class MailJob(Job):
    """
    Job that sends a mail
    """

    def __init__(
        self,
        subject: str,
        content: str,
        mail_address: str = getpass.getuser(),
        wait_on: Optional[Union[Iterable[tk.AbstractPath], tk.AbstractPath]] = None,
        compress: bool = True,
    ):
        """

        :param subject: Subject of the mail
        :param mail_address: Mail address of the user
        :param content:
        :param wait_on:
        """
        self.subject = subject
        self.mail_address = mail_address
        self.content = content
        self.compress = compress

        if wait_on is not None:
            if isinstance(wait_on, tk.AbstractPath):
                self.add_input(wait_on)
            elif isinstance(wait_on, Iterable):
                for wait in wait_on:
                    self.add_input(wait)
            else:
                assert False, "Wrong wait_on input!"

        self.out_content = self.output_path("content")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.compress:
            with gzip.open(self.out_content.get_path(), "w") as f:
                f.write(self.content.encode() + b"\n")
        else:
            with open(self.out_content.get_path(), "w") as f:
                f.write(self.content)

        self.sh("zcat -f {out_content} | mail -s '{subject}' {mail_address}")


class ReportResultsJob(Job):
    """
    Job to report results either via mail and/or just condensed in a file
    """

    def __init__(
        self,
        name: str,
        report: _Report_Type,
        report_format: Callable[[_Report_Type, any], str],
        compress: bool = True,
        **report_format_kwargs,
    ):
        """

        :param name: Name of the report
        :param report: Dictionary containing the report files
        :param report_format: Function that converts the report dictionary to a string
        :param report_format_kwargs: Addtional kwargs for the report format function
        :param compress: Whether to compress the report or not
        """
        self.name = name
        self.report = report
        self.report_format = report_format
        self.report_format_kwargs = report_format_kwargs
        self.compress = compress

        self.sis_command_line = sys.argv
        self.mail_address = getattr(gs, "MAIL_ADDRESS", None)
        self.config_path = tk.config_manager.current_config
        assert self.config_path is not None, "Could not find config path"

        for i in ["date", "user", "name", "config_path", "sis_command_line"]:
            assert i not in report, "%s will be set automatically"

        self.out_report = self.output_path("report.gz")

    def run(self):
        user = getpass.getuser()
        report = self.report.copy()

        report["date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        report["user"] = user
        report["name"] = self.name
        report["config_path"] = self.config_path
        report["sis_command_line"] = str(self.sis_command_line)

        if self.compress:
            with gzip.open(self.out_report.get_path(), "w") as f:
                f.write(
                    self.report_format(report, **self.report_format_kwargs).encode()
                    + b"\n"
                )
        else:
            with open(self.out_report.get_path(), "w") as f:
                f.write(self.report_format(report, **self.report_format_kwargs) + "\n")

        if self.mail_address:
            self.sh(
                "zcat -f {out_report} | mail -s 'Report finished: {name} {config_path}' {mail_address}"
            )

    def tasks(self):
        yield Task("run", mini_task=True)


def gmm_example_report_format(report: _Report_Type) -> str:
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
        if not step_name.startswith("scorer"):
            continue
        if "dev-clean" in step_name:
            set = "dev-clean"
        else:
            set = "dev-other"
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
          Sis Command: {report["sis_command_line"]}

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
