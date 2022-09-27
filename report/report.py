from sisyphus import *

import sys
import getpass
import gzip
from typing import Dict, Union, Callable, Optional
import pprint

_Report_Type = Dict[str, Union[tk.AbstractPath, str]]


class GenerateReportStringJob(Job):
    """
    Job to generate and output a report string
    """

    def __init__(
        self,
        report_values: _Report_Type,
        report_template: Optional[Callable[[_Report_Type], str]] = None,
        compress: bool = True,
    ):
        """

        :param report_values:
        :param report_template:
        :param compress:
        """
        self.report_values = report_values
        self.report_template = report_template
        self.compress = compress

        self.out_report = self.output_path("report.gz" if self.compress else "report")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.report_template:
            report = self.report_template(self.report_values)
        elif callable(self.report_values):
            report = str(self.report_values())
        else:
            report = pprint.pformat(self.report_values, width=140)

        if self.compress:
            with gzip.open(self.out_report.get_path(), "wt") as f:
                f.write(report)
        else:
            with open(self.out_report.get_path(), "wt") as f:
                f.write(report)


class MailJob(Job):
    """
    Job that sends a mail upon completion of an output
    """

    def __init__(
        self,
        result: tk.AbstractPath,
        subject: Optional[str] = None,
        mail_address: str = getpass.getuser(),
        send_contents: bool = False,
    ):
        """

        :param result: graph output that triggers sending the mail
        :param subject: Subject of the mail
        :param mail_address: Mail address of recipient (default: user)
        :param send_contents: send the contents of result in body of the mail
        """
        self.result = result
        self.subject = subject
        self.mail_address = mail_address
        self.send_contents = send_contents

        self.out_status = self.output_var("out_status")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.subject is None:
            subject = f"Output {str(self.result)} is finished"
        else:
            subject = self.subject

        if self.send_contents:
            value = self.sh(
                f"zcat -f {self.result.get_path()} | mail -s '{subject}' {self.mail_address}"
            )
        else:
            value = self.sh(
                f"echo '{subject}' | mail -s '{subject}' {self.mail_address}"
            )

        self.out_status.set(value)


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
