__all__ = ["GetColumnsFromCsvFileJob"]

import csv
import json

from sisyphus import Job, Task, tk

from i6_core import util


class GetColumnsFromCsvFileJob(Job):
    """
    Dumps the values of a given set of columns from a csv file onto separate text files.
    The csv file must have been previously dumped with :funcref:`csv.writer` and :funcref:`csv.writerow`.

    **DISCLAIMER: the output for each column is written via `json.dumps` to escape conflictive characters like newlines.
    `json.dumps` might escape additional tokens of its own, like double quotes.
    The job doesn't do anything in this case, and additional postprocessing is left to the user.**

    The job uses the default `csv.reader` parameters,
    which are expected to be compatible with the default `csv.writer` parameters.
    For instance, the contents of a column can contain the csv delimiter or newlines.
    In this case, the start/end of each cell in a column is properly marked
    (see e.g. https://docs.python.org/3/library/csv.html#csv.Dialect.quotechar).

    The i-th output file contains the i-th column.
    """

    def __init__(self, csv_file: tk.Path, columns: list[int], delimiter: str = ","):
        """
        :param csv_file: Csv file for which to retrieve the corresponding column.
        :param column: Zero-based index for the column for which to retrieve the values.
        :param delimiter: Delimiter of the csv file.
        """
        self.csv_file = csv_file
        self.columns = columns
        self.delimiter = delimiter

        self.out_column_values = {i: self.output_path(f"out_col_{i}.txt.gz") for i in self.columns}

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        # No need to dump the outputs as a csv file, since each output just contains a single column.
        opened_outs = {i: util.uopen(self.out_column_values[i], "wt") for i in self.columns}
        with util.uopen(self.csv_file, "rt") as f_in:
            in_csv = csv.reader(f_in, delimiter=self.delimiter)
            for csv_line in in_csv:
                for column in self.columns:
                    # json.dumps helps escaping conflictive characters, e.g. newlines.
                    escaped_csv_col = json.dumps(csv_line[column], ensure_ascii=False)

                    # Remove artificial start/end quotes added by json.dumps.
                    assert escaped_csv_col.startswith('"') and escaped_csv_col.endswith('"')
                    opened_outs[column].write(f"{escaped_csv_col[1:-1]}\n")
